from typing import Callable, List, Dict, Literal
import math

from hnne import HNNE
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import LabelEncoder


def generate_metric_frames(
    encoded_data: np.array,
    df: pd.DataFrame,
    y: str,
    y_pred: str,
    metric: Callable,
    feature_types: Dict[
        str, Literal["raw", "nominal", "ordinal", "numerical", "embedding"]
    ],
    remove_outliers: bool,
):
    """
    Generate clustering on multiple levels and create data structures for additional processing lateron.

    :param encoded_data: Encoded and normalized feature data for clustering.
    :param df: The dataframe containing ground-truth and predictions.
    :param y: The name of the ground-truth column.
    :param y_pred: The name of the predictions column.
    :param metric: The metric function.
    :param feature_types: The types of all features present in the encoded_data. Used to determine "univariate" case.
    :param remove_outliers: Compute metric in a non-vectorized way for each sample and remove outliers.

    """
    # Special cases, only one or two nominal feature where binning and clustering makes no sense.
    # Encoded data will not be used in this case, instead there will be one clustering with one entry
    # for each category.
    projection = None

    if (
        len(feature_types.values()) == 1
        and list(feature_types.values())[0] == "nominal"
    ):
        clustering_cols = ["clustering_0"]
        clustering_df = pd.DataFrame(
            data=LabelEncoder().fit_transform(df[list(feature_types.keys())[0]]),
            columns=clustering_cols,
            index=df.index,
        )  # TODO: Possible track relation between integers and real categorical values for nicer explanations.
    elif (
        len(feature_types.values()) == 2
        and list(feature_types.values())[0] == "nominal"
        and list(feature_types.values())[1] == "nominal"
    ):
        combinations = (
            df[[list(feature_types.keys())[0], list(feature_types.keys())[1]]]
            .value_counts()
            .index.values
        )
        copied_df = df.copy()
        copied_df["combination_id"] = -1
        for i, combination in enumerate(combinations):
            copied_df.loc[
                (copied_df[list(feature_types.keys())[0]] == combination[0])
                & (copied_df[list(feature_types.keys())[1]] == combination[1]),
                "combination_id",
            ] = i
        clustering_cols = ["clustering_0"]
        clustering_df = pd.DataFrame(
            data=LabelEncoder().fit_transform(copied_df["combination_id"]),
            columns=clustering_cols,
            index=df.index,
        )  # TODO: Possible track relation between integers and real categorical values for nicer explanations.
    else:
        # All other cases are handled here in an implicit way. Clustering is used for binning.
        # Identify hierarchical clustering with h-nne
        # As h-nne fails for less than 2 dimensions introduce dummy dimension in this case
        num_features = encoded_data.shape[1]
        if num_features <= 1:
            encoded_data = np.concatenate(
                (encoded_data, np.zeros((encoded_data.shape[0], 1))), axis=1
            )
        try:
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

        except:
            # The projection might fail if there are not enough data points.
            # In this case just use other clustering approach as fallback.
            print(
                "Warning: Using hierarchical clustering failed. Probably the provided datapoints were not enough. HDBSCAN fallback might yield bad results!"
            )
            hdbscan = HDBSCAN(min_cluster_size=2)
            hdbscan.fit(encoded_data)
            partitions = hdbscan.labels_[..., np.newaxis]
            partition_sizes = [len(np.unique(hdbscan.labels_))]
            partition_levels = len(partition_sizes)
            # TODO: This doesn't necessarily make sense as noisy samples will be treated as own cluster. However that is how it is now.

        clustering_cols = [f"clustering_{i}" for i in range(partition_levels)]
        clustering_df = pd.DataFrame(
            data=partitions, columns=clustering_cols, index=df.index
        )

    # Calculate samplewise metric in order to delete well performing samples from clusters
    # This is done as the clustering does probably contain outliers in some cases that influence the
    # overall metric while most of the cluster is actually fine.
    # E.g. word error rate is 20 for one sample but 0.2 for most others. This should not be marked as issue!
    if remove_outliers == True:
        samplewise_metrics = []
        for idx, sample in df.iterrows():
            sample_metric = metric(np.array([sample[y]]), np.array([sample[y_pred]]))
            samplewise_metrics.append(sample_metric)
        clustering_df["metric"] = samplewise_metrics

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

    return projection, mfs, clustering_df, clustering_cols, clustering_metric_cols


def detect_issues(
    mfs: List[MetricFrame],
    clustering_df: pd.DataFrame,
    clustering_cols: List[str],
    min_drop: float,
    min_support: int,
    n_slices: int,
    criterion: Literal["drop", "support", "drop*support"],
    metric_mode: Literal["min", "max"],
    drop_reference: Literal["overall", "parent"],
    remove_outliers: bool,
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

    print(
        f"Detecting issues for criteria n_slices={n_slices}, criterion={criterion}, min_drop={min_drop}, min_support={min_support}."
    )

    previous_clustering_col = None

    for hierarchy_level, (mf, clustering_col) in enumerate(zip(mfs, clustering_cols)):
        if drop_reference == "overall":
            drop_reference_value = mf.overall.values[0]
        elif drop_reference == "parent":
            if previous_clustering_col is not None:
                drop_reference_value = []
                for c in mf.by_group.index:
                    parent_metric = clustering_df[clustering_df[clustering_col] == c][
                        f"{previous_clustering_col}_metric"
                    ].iloc[0]
                    drop_reference_value.append(parent_metric)
                drop_reference_value = np.array(drop_reference_value)
            else:
                drop_reference_value = mf.overall.values[0]

        else:
            raise RuntimeError(
                "Invalid value for parameter drop_reference. Has to be either overall or parent."
            )

        # Calculate cluster support
        drops = (
            drop_reference_value - mf.by_group.values[:, 0]
            if metric_mode == "max"
            else mf.by_group.values[:, 0] - drop_reference_value
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

        group_df["level"] = hierarchy_level

        # Compute the "true" support for each cluster.
        # There could be clusters where the overall metric looks bad but this is actually caused by one single outlier.
        # Those should not be marked as potential slices if they do not fulfil the min support criterion.
        if remove_outliers:
            true_supports = []
            true_drops = []
            for (cluster_idx, cluster_metric), cluster_drop, cluster_support in zip(
                mf.by_group.iterrows(), drops, supports
            ):
                cluster_metric = cluster_metric["metric"]
                samplewise_metrics = clustering_df[
                    clustering_df[clustering_col] == cluster_idx
                ]["metric"].values
                samplewise_drops = (
                    mf.overall.values[0] - samplewise_metrics
                    if metric_mode == "max"
                    else samplewise_metrics - mf.overall.values[0]
                )
                median_abs_deviation = np.median(
                    np.abs(samplewise_drops - np.median(samplewise_drops))
                )
                valid_samples = (
                    samplewise_drops
                    >= (np.median(samplewise_drops) - 2 * median_abs_deviation)
                ) & (
                    samplewise_drops
                    <= (np.median(samplewise_drops) + 2 * median_abs_deviation)
                )
                true_support = valid_samples.sum()
                true_supports.append(true_support)
                # If there are no valid samples in a slice, treat as having no drop
                if valid_samples.sum() > 0:
                    true_drop = np.mean(samplewise_drops[valid_samples])
                else:
                    true_drop = -math.inf
                true_drops.append(true_drop)

            group_df = pd.concat(
                (
                    group_df,
                    pd.DataFrame(data=true_supports, columns=["true_support"]),
                    pd.DataFrame(data=true_drops, columns=["true_drop"]),
                ),
                axis=1,
            )

        group_dfs.append(group_df)

        previous_clustering_col = clustering_col

    # Now mark issues in the prepared dataframe
    # Define columns to use for filtering for issues
    if remove_outliers:
        drop_col = "true_drop"
        support_col = "true_support"
    else:
        drop_col = "drop"
        support_col = "support"

    # Add issue col to all group_dfs
    for group_df in group_dfs:
        group_df["issue"] = False

    # Create one big dataframe for all groupings
    all_groups_df = pd.concat(group_dfs)

    # Create masks for hard filtering criterions min_support, min_drop
    if min_support is not None:
        min_support_mask = all_groups_df[support_col] >= min_support
    else:
        min_support_mask = np.full(len(all_groups_df), True)

    if min_drop is not None:
        min_drop_mask = all_groups_df[drop_col] >= min_drop
    else:
        min_drop_mask = np.full(len(all_groups_df), True)

    # Mark which clusters are issues according to hard criteria
    all_groups_df["issue"] = min_drop_mask & min_support_mask

    # Sort after criterion if n_slices and criterion are set
    if n_slices is not None and criterion is not None:
        all_groups_df["drop*support"] = (
            all_groups_df[drop_col] * all_groups_df[support_col]
        )
        assert (
            criterion == "support" or criterion == "drop" or criterion == "drop*support"
        )
        all_groups_df = all_groups_df.sort_values(criterion, ascending=False)

    # Mark issues and potentially break if n_lices is exceeded
    marked_issue_idx = 0
    for idx, row in all_groups_df.iterrows():
        if row["issue"] == True:
            group_dfs[int(row["level"])].loc[idx] = True

            marked_issue_idx += 1
            if n_slices is not None and marked_issue_idx >= n_slices:
                break

    return group_dfs
