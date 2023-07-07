from typing import Callable, List, Literal

from hnne import HNNE
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame


def generate_metric_frames(
    encoded_data: np.array, df: pd.DataFrame, y: str, y_pred: str, metric: Callable
):
    """
    Generate clustering on multiple levels and create data structures for additional processing lateron.

    :param encoded_data: Encoded and normalized feature data for clustering.
    :param df: The dataframe containing ground-truth and predictions.
    :param y: The name of the ground-truth column.
    :param y_pred: The name of the predictions column.
    :param metric: The metric function.

    """
    # Identify hierarchical clustering with h-nne
    # As h-nne fails for less than 2 dimenions introduce dummy dimension in this case
    num_features = encoded_data.shape[1]
    if num_features <= 1:
        encoded_data = np.concatenate(
            (encoded_data, np.zeros((encoded_data.shape[0], 1))), axis=1
        )

    hnne = HNNE(
        metric="euclidean"
    )  # TODO Probably explore different settings for hnne. Default of metric is cosine. To determine if this is better choice!
    projection = hnne.fit_transform(encoded_data)
    partitions = hnne.hierarchy_parameters.partitions

    partitions = np.flip(
        partitions, axis=1
    )  # reverse the order of the hierarchy levels, go from coarse to fine
    partition_sizes = hnne.hierarchy_parameters.partition_sizes
    partition_levels = len(partition_sizes)

    clustering_cols = [f"clustering_{i}" for i in range(partition_levels)]
    clustering_df = pd.DataFrame(
        data=partitions, columns=clustering_cols, index=df.index
    )

    # Calculate fairness metrics on the clusters with fairlearn
    mfs = []
    clustering_metric_cols = []
    clustering_count_cols = []
    for col in clustering_cols:
        mf = MetricFrame(
            metrics={"metric": metric},
            y_true=df[y],
            y_pred=df[y_pred],
            sensitive_features=clustering_df[col],
        )
        mfs.append(mf)

        metric_col = f"{col}_metric"
        clustering_metric_cols.append(metric_col)
        clustering_df[metric_col] = np.nan

        count_col = f"{col}_count"
        clustering_count_cols.append(count_col)
        clustering_df[count_col] = np.nan

        for idx, row in mf.by_group.iterrows():
            clustering_df.loc[clustering_df[col] == idx, metric_col] = row["metric"]
            clustering_df.loc[clustering_df[col] == idx, count_col] = (
                clustering_df[col] == idx
            ).sum()
    return mfs, clustering_df, clustering_cols, clustering_metric_cols


def detect_issues(
    mfs: List[MetricFrame],
    clustering_df: pd.DataFrame,
    clustering_cols: List[str],
    min_drop: float,
    min_support: int,
    metric_mode: Literal["min", "max"],
):
    """
    Determine the hierarchy levels that most likely capture real problems, based on the previously
    computed hierarchical clustering.
    :param mfs: List of fairlearn MetricFrames computed on different cluster levels.
    :param clustering_df: The dataframe containing the previously created hierarchical clustering.
    :param clustering_cols: List of columns in clustering_df that correspond to different hierarchy levels.
    :param min_drop: Minimum drop that has to be present so the cluster is marked as problematic.
    :param min_support: Minimum number of samples in one cluster so that it is marked as issue.
    :param metric_mode: Optimization goal for the metric. Can be "min" or "max".

    """
    # Determine the hierarchy levels that most likely capture real problems
    # TODO: Determine if these values should be chosen adaptively, potentially differing on every level
    group_dfs = []

    if min_drop is None:
        min_drop = 0.1 * mfs[0].overall.values[0]
    if min_support is None:
        min_support = round(max(0.0025 * len(clustering_df), 5))
    print(
        f"Using {min_support} as minimum support for determining problematic clusters."
    )
    print(f"Using {min_drop} as minimum drop for determining problematic clusters.")

    previous_group_df = None
    previous_clustering_col = None
    for mf, clustering_col in zip(mfs, clustering_cols):
        # Calculate cluster support
        drops = (
            mf.overall.values[0] - mf.by_group.values
            if metric_mode == "max"
            else mf.by_group.values - mf.overall.values[0]
        )
        supports = [
            (clustering_df[clustering_col] == cluster).sum()
            for cluster in mf.by_group.index
        ]
        group_df = pd.concat(
            (
                mf.by_group,
                pd.DataFrame(data=supports, columns=["support"]),
                pd.DataFrame(data=drops, columns=["drop"]),
            ),
            axis=1,
        )

        # print(group_df)

        group_df["issue"] = False

        group_df.loc[
            (group_df["drop"] > min_drop) & (group_df["support"] > min_support),
            "issue",
        ] = True

        # Unmark parent cluster if drop shows mostly on this level
        if previous_group_df is not None:
            for cluster, row in group_df.iterrows():
                group_entries = clustering_df[clustering_df[clustering_col] == cluster]
                assert (
                    group_entries[previous_clustering_col].values[0]
                    == group_entries[previous_clustering_col].values
                ).all()
                parent_cluster = group_entries[previous_clustering_col].values[0]

                if (
                    row["support"] > min_support and row["drop"] > min_drop
                ):  # TODO Verify this rule makes sense, could cause larger clusters to be discarded because of one also bad subcluster
                    previous_group_df.loc[parent_cluster, "issue"] = False
                else:
                    group_df.loc[cluster, "issue"] = False

        group_dfs.append(group_df)

        previous_group_df = group_df
        previous_clustering_col = clustering_col
    return group_dfs
