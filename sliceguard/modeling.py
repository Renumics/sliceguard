import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from typing import Literal

try:
    from flaml import AutoML
except ImportError:
    raise Warning("Optional dependency required! (pip install sliceguard[AutoML])")

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

    y_preds_raw = automl.predict(encoded_data)
    y_preds = (
        label_encoder.inverse_transform(y_preds_raw)
        if task == "classification"
        else y_preds_raw
    )

    return y_preds
