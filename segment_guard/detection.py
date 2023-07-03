from typing import Callable

from hnne import HNNE
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame

def generate_metric_frames(encoded_data: np.array, df: pd.DataFrame, y: str, y_pred: str, metric:Callable):
    """
    Generate clustering on multiple levels and create data structures for additional processing lateron.

    :param encoded_data: Encoded and normalized feature data for clustering.
    :param df: The dataframe containing ground-truth and predictions.
    :param y: The name of the ground-truth column.
    :param y_pred: The name of the predictions column.
    :param metric: The metric function.

    """
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
    

def detect_issues():
     pass