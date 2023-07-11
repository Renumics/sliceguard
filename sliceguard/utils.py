from typing import List, Dict, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
import umap

from .embedding_utils import (
    generate_image_embeddings,
    generate_audio_embeddings,
    generate_text_embeddings,
)


def infer_feature_types(
    features,
    given_feature_types: Dict[
        str, Literal["raw", "nominal", "ordinal", "numerical", "embedding"]
    ],
    precomputed_embeddings: Dict[str, np.array],
    df: pd.DataFrame,
):
    """
    Infer the datatypes of certain features based on their column data.
    :param features: The features that have to be inferred.
    :param given_feature_types: Feature types that are already defined by the user.
    :param df: The dataframe containing all the data.
    """

    feature_types = {}
    for col in features:
        # check if the column is supplied in precomputed embeddings, then always use embedding feature type
        if col in precomputed_embeddings:
            feature_types[col] = "embedding"
            continue

        col_dtype = df[col].dtype

        if col_dtype == "object" and col not in given_feature_types:
            num_unique_values = len(df[col].unique())
            if num_unique_values / len(df) > 0.5:
                print(
                    f"Feature {col} was inferred as referring to raw data. If this is not the case, please specify in feature_types!"
                )
                feature_types[col] = "raw"
            else:
                print(
                    f"Feature {col} was inferred as being categorical. Will be treated as nominal by default. If ordinal specify in feature_types and feature_orders!"
                )
                feature_types[col] = "nominal"
        elif col not in given_feature_types:
            print(
                f"Feature {col} will be treated as numerical value. You can override this by specifying feature_types."
            )
            feature_types[col] = "numerical"
        else:
            assert given_feature_types[col] in (
                "raw",
                "nominal",
                "ordinal",
                "numerical",
                "embedding",
            )
            feature_types[col] = given_feature_types[col]
    return feature_types


def encode_normalize_features(
    features: List[str],
    feature_types: Dict[str, Literal["raw", "nominal", "ordinal", "numerical"]],
    feature_orders: Dict[str, list],
    precomputed_embeddings: Dict[str, np.array],
    embedding_models: Dict[str, str],
    hf_auth_token: str,
    df: pd.DataFrame,
):
    """
    :param features: Names of features that should be encoded and normalized for later processing.
    :param feature_types: The previously inferred or given types of the respective features.
    :param feature_orders: If ordinal features are present you have to supply an order for each of them.
    :param precomputed_embeddings: Precomputed embeddings that the user might supply.
    :param df: The dataframe containing all the data.
    """
    encoded_data = np.zeros((len(df), 0))
    prereduced_embeddings = {}
    raw_embeddings = {}
    for col in features:
        feature_type = feature_types[col]
        if feature_type == "numerical":
            # TODO: Think about proper scaling method. Intuition here is to preserve outliers,
            # however the range of the data can possibly dominate one hot and ordinal encoded features.
            normalized_data = RobustScaler(quantile_range=(2.5, 97.5)).fit_transform(
                df[col].values.reshape(-1, 1)
            )

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
        elif feature_type == "raw" or feature_type == "embedding":
            # Print model that will be used for computing embeddings
            if col in df.columns and col not in precomputed_embeddings:
                model_name_param = (
                    {"model_name": embedding_models[col]}
                    if col in embedding_models
                    else {}
                )
                if "model_name" in model_name_param:
                    print(
                        f"Using {model_name_param['model_name']} for computing embeddings for feature {col}."
                    )
                else:
                    print(
                        f"Using default model for computing embeddings for feature {col}."
                    )
            # Set first entry as for checking type of raw data.
            if col in df.columns:
                first_entry = df[col].iloc[0]
            if col in precomputed_embeddings:  # use precomputed embeddings when given
                embeddings = precomputed_embeddings[col]
                assert len(embeddings) == len(df)
            elif first_entry.lower().endswith(
                ".wav"
            ):  # TODO: Improve data type inference for raw data
                embeddings = generate_audio_embeddings(
                    df[col].values, **model_name_param
                )
                raw_embeddings[col] = embeddings
            elif (
                first_entry.lower().endswith(".jpg")
                or first_entry.lower().endswith(".jpeg")
                or first_entry.lower().endswith(".png")
            ):
                embeddings = generate_image_embeddings(
                    df[col].values, **model_name_param
                )
                raw_embeddings[col] = embeddings
            else:  # Treat as text if nothing known
                embeddings = generate_text_embeddings(
                    df[col].values, **model_name_param
                )
                raw_embeddings[col] = embeddings

            # TODO: Potentially filter out entries without valid embedding or replace with mean?
            reduced_embeddings = umap.UMAP(
                # n_neighbors=min(len(df) - 1, 30),
                # min_dist=0.0,
                n_components=2,
                random_state=42,
            ).fit_transform(embeddings)

            # TODO: Check if normalization makes sense. Probably do not normalize dimensiosn indenpendently!
            # reduced_embeddings = RobustScaler(
            #     quantile_range=(2.5, 97.5)
            # ).fit_transform(
            #     reduced_embeddings
            # )

            # safe this as it can be used for generating explanations again
            # do not normalize as this will probably cause non blobby clusters and it is unclear what clustering assumes
            prereduced_embeddings[col] = reduced_embeddings

            encoded_data = np.concatenate((encoded_data, reduced_embeddings), axis=1)

        else:
            raise RuntimeError(
                "Encountered unknown feature type when encoding and normalizing features."
            )

    return encoded_data, prereduced_embeddings, raw_embeddings
