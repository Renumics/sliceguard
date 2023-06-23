import logging
from typing import List, Literal, Dict, Callable

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from fairlearn.metrics import MetricFrame
from hnne import HNNE
from renumics import spotlight

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
        min_support = None,
        min_drop = None
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

        df = data[features + [y_pred, y]]

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
        hnne = HNNE(
            metric="euclidean"
        )  # TODO Probably explore different settings for hnne. Default of metric is cosine. To determine if this is better choice!
        projection = hnne.fit_transform(encoded_data)
        partitions = np.flip(
            hnne.hierarchy_parameters.partitions, axis=1
        )  # reverse the order of the hierarchy levels, go from coarse to fine
        partition_sizes = hnne.hierarchy_parameters.partition_sizes
        partition_levels = len(partition_sizes)

        clustering_cols = [f"clustering_{i}" for i in range(partition_levels)]
        clustering_df = pd.DataFrame(
            data=partitions,
            columns=clustering_cols,
        )

        # Calculate metrics on per sample level for later use in hierarchy merging
        metrics = []
        for _, row in df.iterrows():
            sample_metric = metric([row[y]], [row[y_pred]])
            metrics.append(sample_metric)
            
        clustering_df["metric"] = np.nan
        clustering_df["metric"] = metrics


        # Calculate fairness metrics on the clusters with fairlearn
        mfs = []
        clustering_metric_cols = []
        clustering_count_cols = []
        for col in clustering_cols:
            mf = MetricFrame(metrics={"metric": metric}, y_true=df[y], y_pred=df[y_pred], sensitive_features=clustering_df[col])
            mfs.append(mf)

            metric_col = f"{col}_metric"
            clustering_metric_cols.append(metric_col)
            clustering_df[metric_col] = np.nan

            count_col = f"{col}_count"
            clustering_count_cols.append(count_col)
            clustering_df[count_col] = np.nan

            for idx, row in mf.by_group.iterrows():
                clustering_df.loc[clustering_df[col] == idx, metric_col] = row["metric"]
                clustering_df.loc[clustering_df[col] == idx, count_col] = (clustering_df[col] == idx).sum()
        
        # Determine the hierarchy levels that most likely capture real problems
        # TODO: Determine if these values should be chosen adaptively, potentially differing on every level
        group_dfs = []

        if min_drop is None:
            min_drop = 0.1 * mfs[0].overall.values[0]
        if min_support is None:
            min_support = round(max(0.0025 * len(df), 5))
        print(f"Using {min_support} as minimum support for determining problematic clusters.")
        print(f"Using {min_drop} as minimum drop for determining problematic clusters.")

        previous_group_df = None
        previous_clustering_col = None
        for mf, clustering_col in zip(mfs, clustering_cols):
            # Calculate cluster support
            drops = mf.overall.values[0] - mf.by_group.values if metric_mode == "max" else mf.by_group.values - mf.overall.values[0]
            supports = [(clustering_df[clustering_col] == cluster).sum() for cluster in mf.by_group.index]
            group_df = pd.concat((mf.by_group, pd.DataFrame(data=supports, columns=["support"]), pd.DataFrame(data=drops, columns=["drop"])), axis=1)

            
            group_df["issue"] = False

            group_df.loc[(group_df["drop"] > min_drop) & (group_df["support"] > min_support), "issue"] = True
            
            # Check overlap with parent cluster and calculate how much drop this cluster causes
            # Unmark parent if drop shows mostly on this level
            if previous_group_df is not None:
                for cluster, row in group_df.iterrows():
                    group_entries = clustering_df[clustering_df[clustering_col] == cluster]
                    assert (group_entries[previous_clustering_col].values[0] == group_entries[previous_clustering_col].values).all()
                    parent_cluster = group_entries[previous_clustering_col].values[0]
                    parent_cluster_info = previous_group_df.loc[parent_cluster]

                    num_child_clusters = len(clustering_df[clustering_df[previous_clustering_col] == parent_cluster][clustering_col].unique())



                    parent_group_entries = clustering_df[clustering_df[previous_clustering_col] == parent_cluster]
                    abs_parent_drop = (clustering_df.loc[parent_group_entries.index]["metric"] - mf.overall.values[0]).sum()

                    abs_cluster_drop = (clustering_df.loc[group_entries.index]["metric"] - mf.overall.values[0]).sum()

                    if (abs_cluster_drop / abs_parent_drop) > (1 / num_child_clusters): # TODO Verify this rule makes sense and the factor is okay probably use 2*std instead
                        previous_group_df.loc[parent_cluster, "issue"] = False
                        print("----------------------------------------")
                        print((abs_cluster_drop / abs_parent_drop))
                        print("cluster")
                        print(abs_cluster_drop)
                        print("parent")
                        print(abs_parent_drop)
                    else:
                        group_df.loc[cluster, "issue"] = False

            group_dfs.append(group_df)

            previous_group_df = group_df
            previous_clustering_col = clustering_col
        
        num_issues = np.sum([group_df["issue"].sum() for group_df in group_dfs])

        print(f"Identified {num_issues} problematic segments.")
        for hierarchy_level, group_df in enumerate(group_dfs):
            print(f"-------- Hierarchy level {hierarchy_level} --------")
            print(group_df[group_df["issue"] == True])


        # spotlight.show(clustering_df)

    def report(self):
        """
        Create an interactive report on the found issues in spotlight.
        """
        pass
        # spotlight.show(issues=[])
