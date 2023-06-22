import logging
from typing import List, Literal, Dict, Callable

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler


class SegmentGuard:
    """
    The main class for detecting issues in your data
    """

    def find_issues(
        self,
        data: pd.DataFrame,
        features: List[str],
        y: str,
        y_pred: str,
        metric: Callable,
        metric_mode: Literal["min", "max"] = "max",
        feature_types: Dict[
            str, Literal["raw", "nominal", "ordinal", "numerical"]
        ] = {},
        feature_orders: Dict[str, list] = {},
    ):
        """
        Find segments that are classified badly by your model.

        :param data: A pandas dataframe containing your data.
        :param features: A list of columns that contains features to feed into your model but also metadata.
        :param y: The column containing the ground-truth label.
        :param y_pred: The column containing your models prediction.
        :param metric: A callable metric function that must correspond to the form metric(y_true, y_pred) -> scikit-learn style.
        :param metric_mode: What do you optimize your metric for? max is the right choice for accuracy while e.g. min is good for regression error.
        :param feature_types: Specify how your feature should be treated in encoding and normalizing.
        :param feature_orders: If your feature is ordinal, specify the order of that should be used for encoding. This is required for EVERY ordinal feature.

        """

        df = data[features]

        # Try to infer the column dtypes
        dataset_length = len(df)

        for col in features:
            col_dtype = df[col].dtype

            if col_dtype == "object" and col not in feature_types:
                num_unique_values = len(df[col].unique())
                if num_unique_values / dataset_length > 0.5:
                    logging.warning(
                        f"Feature {col} was inferred as referring to raw data. If this is not the case, please specify in feature_types!"
                    )
                    feature_types[col] = "raw"
                else:
                    logging.warning(
                        f"Feature {col} was inferred as being categorical. Will be treated as nominal by default. If ordinal specify in feature_types and feature_orders!"
                    )
                    feature_types[col] = "nominal"
            elif col not in feature_types:
                logging.warning(
                    f"Feature {col} will be treated as numerical value. You can override this by specifying feature_types."
                )
                feature_types[col] = "numerical"
            else:
                assert feature_types[col] in ("raw", "nominal", "ordinal", "numerical")

        print(feature_types)

        # TODO: Potentially also explicitely check for univariate and bivariate fairness issues, however start with the more generic variant

        # Encode the features for clustering according to inferred types
        encoded_data = np.zeros((len(df), 0))
        for col in features:
            feature_type = feature_types[col]
            if feature_type == "numerical":
                # TODO: Think about proper scaling method. Intuition here is to preserve outliers,
                # however the range of the data can possibly dominate one hot and ordinal encoded features.
                normalized_data = RobustScaler(
                    quantile_range=(2.5, 97.5)
                ).fit_transform(df[col].values.reshape(-1, 1))

                encoded_data = np.concatenate((encoded_data, normalized_data), axis=1)
            elif feature_type == "nominal":
                one_hot_data = OneHotEncoder(sparse_output=False).fit_transform(
                    df[col].values.reshape(-1, 1)
                )
                encoded_data = np.concatenate((encoded_data, one_hot_data), axis=1)
            elif feature_type == "ordinal":
                if col not in feature_orders:
                    raise RuntimeError(
                        "All ordinal features need a specified order! Use feature_orders parameter."
                    )
                feature_order = feature_orders[col]
                unique_categories = df[col].unique()
                category_difference = np.setdiff1d(unique_categories, feature_order)
                if len(category_difference) > 0:
                    raise RuntimeError(
                        f"For ordinal features EACH category has to occur in the specified order. Missing {category_difference}."
                    )
                ordinal_data = OrdinalEncoder(categories=[feature_order]).fit_transform(
                    df[col].values.reshape(-1, 1)
                )
                ordinal_data = ordinal_data / (
                    len(feature_order) - 1
                )  # normalize with unique category count to make compatible with range of one hot encoding
                encoded_data = np.concatenate((encoded_data, ordinal_data), axis=1)
            elif feature_type == "raw":
                raise RuntimeError(
                    "Not implemented yet. Later embeddings will be generated to make this work on unstructured data."
                )
            else:
                raise RuntimeError(
                    "Encountered unknown feature type when encoding and normalizing features."
                )

        # Perform detection of problematic clusters based on the given features
        # 1. A hierarchical clustering is performed and metrics are calculated for all hierarchies
        # 2. hierarchy level that is most indicative of a real problem is then determined
        # 3. the reason for the problem e.g. feature combination or rule that is characteristic for the cluster is determined.

        # Identify hierarchical clustering with h-nne
        

    def report(self):
        """
        Create an interactive report on the found issues in spotlight.
        """
        pass
        # spotlight.show(issues=[])
