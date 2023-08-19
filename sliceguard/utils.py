from typing import List, Dict, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances
import umap

from .embeddings import (
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
    hf_num_proc: Optional[int],
    hf_batch_size: int,
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
            one_hot_data = (
                one_hot_data / 1.41
            )  # all other data shold have approximately range 0 to 1 so match this!
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
                hf_model_params = (
                    {
                        "model_name": embedding_models[col],
                        "hf_auth_token": hf_auth_token,
                        "hf_num_proc": hf_num_proc,
                        "hf_batch_size": hf_batch_size,
                    }
                    if col in embedding_models
                    else {}
                )
                if "model_name" in hf_model_params:
                    print(
                        f"Using {hf_model_params['model_name']} for computing embeddings for feature {col}."
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
                raw_embeddings[
                    col
                ] = embeddings  # also save them here as they are used in report
            elif first_entry.lower().endswith(".wav") or first_entry.lower().endswith(
                ".mp3"
            ):  # TODO: Improve data type inference for raw data
                embeddings = generate_audio_embeddings(
                    df[col].values, **hf_model_params
                )
                raw_embeddings[col] = embeddings
            elif (
                first_entry.lower().endswith(".jpg")
                or first_entry.lower().endswith(".jpeg")
                or first_entry.lower().endswith(".png")
            ):
                embeddings = generate_image_embeddings(
                    df[col].values, **hf_model_params
                )
                raw_embeddings[col] = embeddings
            else:  # Treat as text if nothing known
                embeddings = generate_text_embeddings(df[col].values, **hf_model_params)
                raw_embeddings[col] = embeddings

            # TODO: Potentially filter out entries without valid embedding or replace with mean?
            is_all_embeddings = all(
                v == "embedding" or v == "raw" for v in feature_types.values()
            )
            num_embedding_dimensions = 64
            num_mixed_dimensions = 8
            if is_all_embeddings:
                print(
                    f"All supplied features are raw data or embeddings respectively. They will be reduced to vectors of {num_embedding_dimensions} for computational efficiency."
                )
            else:
                # TODO: Check if also using cosine distance could be an additional measure or an alternative to complicated normalization.
                print(
                    f"The supplied features are of mixed type. In order to provide better clustering results embeddings will be pre-reduced to {num_mixed_dimensions} and normalized."
                )

            reduced_embeddings = umap.UMAP(
                n_neighbors=min(embeddings.shape[0] - 1, 20),
                n_components=min(
                    embeddings.shape[0] - 2,
                    num_embedding_dimensions
                    if is_all_embeddings
                    else num_mixed_dimensions,
                ),  # TODO: Do not hardcode this, probably determine based on embedding size and variance. Also, check implications on normalization.
                # min_dist=0.0,
                random_state=42,
            ).fit_transform(embeddings)

            # Do a normalization of the reduced embedding to match one hot encoded and ordinal encoding respectively
            # Therefore we will run hdbscan on the data real quick to do an estimate of the cluster distances.
            # Then the data will be normalized to make the average cluster distances approximately 1.
            if not is_all_embeddings:
                hdbscan = HDBSCAN(
                    min_cluster_size=2, metric="euclidean", store_centers="centroid"
                )
                hdbscan.fit(reduced_embeddings)
                centroids = hdbscan.centroids_
                distances = pairwise_distances(centroids, centroids, metric="euclidean")
                mean_distance = distances.flatten().mean()
                reduced_embeddings = reduced_embeddings / mean_distance

            # safe this as it can be used for generating explanations again
            # do not normalize as this will probably cause non blobby clusters and it is unclear what clustering assumes
            prereduced_embeddings[col] = reduced_embeddings

            encoded_data = np.concatenate((encoded_data, reduced_embeddings), axis=1)

        else:
            raise RuntimeError(
                "Encountered unknown feature type when encoding and normalizing features."
            )

    return encoded_data, prereduced_embeddings, raw_embeddings
