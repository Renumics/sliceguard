import datasets
from functools import partial

def _check_training_imports():
    try:
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            DefaultDataCollator,
            TrainingArguments,
            Trainer,
        )
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


def load_model(model_name, num_labels):
    """Load the pre-trained model with AutoModelForImageClassification. Specify number of labels."""
    from transformers import AutoModelForImageClassification

    checkpoint = model_name
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint, num_labels=num_labels
    )
    return model


def finetune_image_classfier(
    df,
    model_name="google/vit-base-patch16-224-in21k",
    output_model_folder="model_folder",
    epochs=5,
):
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
        # callbacks=[PrinterCallback],
    )

    train_results = trainer.train()
    model.save_pretrained(output_model_folder)
    image_processor.save_pretrained(output_model_folder)