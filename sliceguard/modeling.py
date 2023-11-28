from typing import Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from typing import Literal
from .models.huggingface import (
    finetune_image_classifier,
    generate_image_pred_probs_embeddings,
)


def get_automl_imports():
    try:
        from flaml import AutoML
        import xgboost
        import shap
    except ImportError:
        raise RuntimeError(
            'Optional dependencies shap, flaml and xgboost required! (run pip install "sliceguard[automl]")'
        )

    return AutoML, shap


def fit_outlier_detection_model(encoded_data: np.array):
    clf = IsolationForest()
    clf.fit(encoded_data)
    outlier_scores = -clf.score_samples(encoded_data)
    return outlier_scores


def fit_classification_regression_model(
    df: pd.DataFrame,
    y_column: str,
    feature_types: Dict[str, str],
    raw_feature_types: Dict[str, str],
    encoded_data: np.array,
    task: Literal["classification", "regression"],
    split: np.array = None,
    train_split: str = None,
    time_budget: float = 20.0,
    hf_model=None,
    hf_model_architecture=None,
    hf_model_output_dir=None,
    hf_model_epochs=5,
    hf_auth_token=None,
):
    if task == "classification":
        label_encoder = LabelEncoder()
        encoded_ys = label_encoder.fit_transform(df[y_column].values)
        num_classes = len(label_encoder.classes_)
        if num_classes > 2:
            automl_metric = "roc_auc_ovr"
        elif num_classes == 2:
            automl_metric = "roc_auc"
        else:
            raise RuntimeError("Invalid number of classes. Must be more than one.")
    else:
        encoded_ys = df[y_column].values
        num_classes = None
        label_encoder = None
        automl_metric = "mse"

    if (
        task == "classification"
        and hf_model is not None
        and hf_model_architecture is not None
        and hf_model_output_dir is not None
        and len(list(raw_feature_types.values())) == 1
        and list(raw_feature_types.values())[0] == "image"
    ):
        y_probs, y_preds, classes = _fit_hf_model_image_classification(
            df,
            list(raw_feature_types.keys())[0],
            y_column,
            hf_model,
            hf_model_architecture,
            hf_model_output_dir,
            split,
            train_split,
            label_encoder,
            encoded_ys,
            hf_model_epochs,
            hf_auth_token=hf_auth_token,
        )
        model = None
    else:
        y_probs, y_preds, classes, model = _fit_embedding_based_model(
            encoded_data,
            task,
            split,
            train_split,
            time_budget,
            label_encoder,
            encoded_ys,
            automl_metric,
        )

    return y_preds, y_probs, classes, model


def _fit_hf_model_image_classification(
    df,
    image_column,
    label_column,
    model_name,
    model_architecture,
    model_output_dir,
    split,
    train_split,
    label_encoder,
    encoded_ys,
    epochs,
    hf_auth_token,
):
    train_df = pd.DataFrame()
    train_df["image"] = df[image_column]
    train_df["label"] = df[label_column]
    train_df = train_df.rename(
        columns={image_column: "image", label_column: "label"}
    )  # This might not work if the columns are already present.
    train_df["label"] = encoded_ys
    if split is not None or train_split is not None:
        print(
            "Warning: The Huggingface model finetuning does not yet care for any split arguments."
        )
    finetune_image_classifier(
        train_df,
        model_name=model_name,
        model_architecture=model_architecture,
        output_model_folder=model_output_dir,
        hf_auth_token=hf_auth_token,
    )
    y_preds_raw, probs, _ = generate_image_pred_probs_embeddings(
        train_df["image"], model_name=model_output_dir, return_embeddings=False
    )
    y_preds = label_encoder.inverse_transform(y_preds_raw)

    classes = label_encoder.classes_
    return np.array(probs), np.array(y_preds), classes


def _fit_embedding_based_model(
    encoded_data,
    task,
    split,
    train_split,
    time_budget,
    label_encoder,
    encoded_ys,
    automl_metric,
):
    AutoML, _ = get_automl_imports()

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
    return y_probs, y_preds, classes, automl
