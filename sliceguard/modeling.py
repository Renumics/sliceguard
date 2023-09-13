from functools import partial
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from typing import Literal
import datasets

def get_automl_imports():
    try:
        from flaml import AutoML
        import xgboost
    except ImportError:
        raise RuntimeError(
            'Optional dependencies flaml and xgboost required! (run pip install "sliceguard[automl]")'
        )

    return AutoML


def fit_outlier_detection_model(encoded_data: np.array):
    clf = IsolationForest()
    clf.fit(encoded_data)
    outlier_scores = -clf.score_samples(encoded_data)
    return outlier_scores


def fit_classification_regression_model(
    encoded_data: np.array,
    ys: np.array,
    task: Literal["classification", "regression"],
    split: np.array = None,
    train_split: str = None,
    time_budget: float = 20.0,
):
    if task == "classification":
        label_encoder = LabelEncoder()
        encoded_ys = label_encoder.fit_transform(ys)
        num_classes = len(label_encoder.classes_)
        if num_classes > 2:
            automl_metric = "roc_auc_ovr"
        elif num_classes == 2:
            automl_metric = "roc_auc"
        else:
            raise RuntimeError("Invalid number of classes. Must be more than one.")
    else:
        encoded_ys = ys
        num_classes = None
        automl_metric = "mse"

    AutoML = get_automl_imports()

    if split is not None:
        if train_split is not None:
            split_mask = split == train_split

            train_ys = encoded_ys[split_mask]

            automl = AutoML()
            automl.fit(
                encoded_data[split_mask],
                train_ys,
                task=task,
                metric=automl_metric,
                estimator_list=["xgboost"],
                time_budget=time_budget,
            )
        else:
            if len(np.unique(split)) < 2:
                raise RuntimeError(
                    "Split column must contain at least 2 separate classes!"
                )

            automl_settings = {
                "eval_method": "cv",
                "groups": split,
                "split_type": "group",
                "task": task,
                "metric": automl_metric,
                "estimator_list": ["xgboost"],
                "time_budget": time_budget,
                "n_splits": min(len(np.unique(split)), 5),
            }

            automl = AutoML()
            automl.fit(encoded_data, encoded_ys, **automl_settings)

    else:
        automl = AutoML()
        automl.fit(
            encoded_data,
            encoded_ys,
            task=task,
            metric=automl_metric,
            estimator_list=["xgboost"],
            time_budget=time_budget,
        )

    if task == "classification":
        y_probs = automl.predict_proba(encoded_data)
        y_preds_raw = np.argmax(y_probs, axis=1)
        y_preds = label_encoder.inverse_transform(y_preds_raw)
        classes = label_encoder.classes_
    elif task == "regression":
        y_preds = automl.predict(encoded_data)
        y_probs = None
        classes = None
    else:
        raise RuntimeError("Could not run inference. Not valid task given.")

    return y_preds, y_probs, classes




def _check_training_imports():
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification, DefaultDataCollator,  TrainingArguments, Trainer
        import torch
    except ImportError:
        raise Warning(
            'Optional dependency required! (pip install "sliceguard[embedding]")'
        )

    return 

def _transform(example_batch, image_processor):
    inputs = image_processor(
        [x.convert("RGB") for x in example_batch["image"]], return_tensors="pt"
    )
    inputs["label"] = example_batch["label"]
    return inputs

def load_image_preprocessor(model_name):
    """Load an image preprocessor for the model. This will resize the images to the correct size for the model."""
    from transformers import AutoImageProcessor
    checkpoint = model_name
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    return image_processor

def load_model(model_name,  num_labels):
    """Load the pre-trained model with AutoModelForImageClassification. Specify number of labels."""
    from transformers import AutoModelForImageClassification
    checkpoint = model_name
    model = AutoModelForImageClassification.from_pretrained(checkpoint, num_labels=num_labels)
    return model

def finetune_image_classfier(df, model_name="google/vit-base-patch16-224-in21k", output_model_folder="model_folder", epochs=5):
    _check_training_imports()
    import torch
    from transformers import DefaultDataCollator, TrainingArguments, Trainer

    ds = datasets.Dataset.from_pandas(df)

    image_processor = load_image_preprocessor(model_name)
    transform_with_processor = partial(_transform, image_processor=image_processor)
    prepared_ds = ds.cast_column("image", datasets.Image())
    prepared_ds = prepared_ds.with_transform(transform_with_processor)
    model = load_model(model_name, num_labels=df["label"].nunique())

    training_args = TrainingArguments(
        output_model_folder,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,  # use 0.04 for testing with a few frames. Use higher values for longer movies
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=prepared_ds,
        eval_dataset=prepared_ds,
        tokenizer=image_processor,
        #callbacks=[PrinterCallback],
    )

    train_results = trainer.train()
    model.save_pretrained(output_model_folder)
    image_processor.save_pretrained(output_model_folder)