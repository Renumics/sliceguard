import logging
from typing import List, Literal, Dict, Callable

import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame
from hnne import HNNE
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

from renumics import spotlight
from renumics.spotlight.analysis.typing import DataIssue

from .utils import infer_feature_types, encode_normalize_features


class SegmentGuard:
    """
    The main class for detecting issues in your data
    """

    def generate_summary_report(data, raw_data, precomputed_embeddings, features, metadata):
        """
        Generate a report on biases, potential underrepresented populations, problematic features, hidden stratification etc.
        """
        raise RuntimeError("This functionality has not been implemented yet.")

    # TODO: Introduce control features to account for expected variations
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
        precomputed_embeddings: Dict[str, np.array] = {},
        min_support: int=None,
        min_drop: float=None,
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
        :param precomputed_embeddings: Supply precomputed embeddings for raw columns. E.g. if repeatedly running checks on your data.
        :min_support: Minimum support for clusters that are listed as issues. If you are more looking towards outliers choose small values, if you target biases choose higher values.
        :min_drop: Minimum metric drop a cluster has to have to be counted as issue compared to the result on the whole dataset.
        """

        assert (all([f in data.columns for f in features])) and (y in data.columns) and (y_pred in data.columns) # check presence of all columns
        df = data  # just rename the variable for shorter naming

        # Try to infer the column dtypes
        feature_types = infer_feature_types(features, feature_types, df)

        # TODO: Potentially also explicitely check for univariate and bivariate fairness issues, however start with the more generic variant

        # Encode the features for clustering according to inferred types
        encoded_data, prereduced_embeddings, raw_embeddings = encode_normalize_features(features, feature_types, feature_orders, precomputed_embeddings, df)

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

        # Determine the hierarchy levels that most likely capture real problems
        # TODO: Determine if these values should be chosen adaptively, potentially differing on every level
        group_dfs = []

        if min_drop is None:
            min_drop = 0.1 * mfs[0].overall.values[0]
        if min_support is None:
            min_support = round(max(0.0025 * len(df), 5))
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
                    group_entries = clustering_df[
                        clustering_df[clustering_col] == cluster
                    ]
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

        num_issues = np.sum([group_df["issue"].sum() for group_df in group_dfs])

        print(f"Identified {num_issues} problematic segments.")

        issue_df = pd.DataFrame(data=[-1] * len(df), columns=["issue"], index=df.index)
        issue_df["issue"] = issue_df["issue"].astype(int)
        issue_df["issue_metric"] = np.nan

        issue_index = 0
        for _, (group_df, clustering_col, clustering_metric_col) in enumerate(
            zip(group_dfs, clustering_cols, clustering_metric_cols)
        ):
            hierarchy_issues = group_df[group_df["issue"] == True].index
            for issue in hierarchy_issues:
                issue_indices = clustering_df[
                    clustering_df[clustering_col] == issue
                ].index.values
                issue_df.loc[issue_indices, "issue"] = issue_index
                issue_metric = clustering_df[clustering_df[clustering_col] == issue][
                    clustering_metric_col
                ].values[0]
                issue_df.loc[issue_indices, "issue_metric"] = issue_metric
                issue_index += 1

        # Derive rules that are characteristic for each identified problem segment
        # This is done to help understanding of the problem reason
        # First stage this will be only importance values!

        # Encode the data and keep track of conversions to keep interpretable
        feature_groups = []  # list of indices for grouped features
        current_feature_index = 0
        classification_data = np.zeros((len(df), 0))
        label_encoders = {}
        for col in features:
            feature_type = feature_types[col]
            if feature_type == "numerical":
                classification_data = np.concatenate(
                    (classification_data, df[col].values.reshape(-1, 1)), axis=1
                )
                feature_groups.append([current_feature_index])
                current_feature_index += 1
            elif feature_type == "nominal" or feature_type == "ordinal":
                label_encoder = LabelEncoder()
                integer_encoded_data = label_encoder.fit_transform(
                    df[col].values
                ).reshape(-1, 1)
                label_encoders[col] = label_encoder
                classification_data = np.concatenate(
                    (classification_data, integer_encoded_data), axis=1
                )
                feature_groups.append([current_feature_index])
                current_feature_index += 1
            elif feature_type == "raw":
                reduced_embeddings = prereduced_embeddings[col]
                classification_data = np.concatenate(
                    (classification_data, reduced_embeddings), axis=1
                )

                feature_groups.append(
                    list(
                        range(
                            current_feature_index,
                            current_feature_index + reduced_embeddings.shape[1],
                        )
                    )
                )
                current_feature_index += reduced_embeddings.shape[1]
            else:
                raise RuntimeError(
                    "Met unexpected feature type while generating explanations."
                )

        # Fit tree to generate feature importances
        # TODO: Potentially replace with simpler univariate mechanism, see also spotlight relevance score
        # TODO: Probably try shap or something similar
        issue_df["issue_explanation"] = ""

        for issue in issue_df["issue"].unique():
            if issue == -1:  # Skip data points with no issues
                continue
            issue_indices_pandas = issue_df[issue_df["issue"] == issue].index
            issue_indices_list = np.where(issue_df["issue"] == issue)[0]
            y = np.zeros(len(issue_df))
            y[issue_indices_list] = 1
            clf = DecisionTreeClassifier(
                max_depth=3, max_features=4
            )  # keep the trees simple to not overfit
            clf.fit(classification_data, y)

            preds = clf.predict(classification_data)
            f1 = f1_score(y, preds)

            importances = clf.feature_importances_

            # aggregate importances of grouped features
            agg_importances = []
            for feature_group in feature_groups:
                if len(feature_group) > 1:
                    agg_importances.append(importances[feature_group].sum())
                else:
                    agg_importances.append(importances[feature_group[0]])
            importances = np.array(agg_importances)

            feature_order = np.argsort(importances)[::-1]
            ordered_importances = importances[feature_order]
            ordered_features = np.array(features)[feature_order]

            # if f1 > 0.7: # only add explanation if it is succicient to classify cluster?
            importance_strings = []
            for f, i in zip(ordered_features[:3], ordered_importances[:3]):
                importance_strings.append(f"{f}, ({i:.2f})")
            issue_df.loc[issue_indices_pandas, "issue_explanation"] = ", ".join(
                importance_strings
            )
        self._issue_df = issue_df
        self._metric_mode = metric_mode
        self._df = df
        self.embeddings = raw_embeddings

        return issue_df

        # df["age"] = df["age"].astype("category")
        # df["gender"] = df["gender"].astype("category")
        # df["accent"] = df["accent"].astype("category")
        # spotlight.show(df, wait=True)

    def report(self, spotlight_dtype: Dict[str, spotlight.dtypes.base.DType]={}):
        """
        Create an interactive report on the found issues in spotlight.
        :param spotlight_dtype: Define a datatype mapping for the interactive spotlight report. Will be passed to dtypes parameter of spotlight.show.
        """
        # Some basic checks
        assert self._issue_df is not None
        assert len(self._df) == len(self._issue_df)
        assert all(self._df.index == self._issue_df.index)
        

        df = pd.concat((self._df, self._issue_df), axis=1)

        data_issue_severity = []
        data_issues = []
        for issue in self._issue_df["issue"].unique():
            if issue == -1:
                continue
            issue_rows = np.where(self._issue_df["issue"] == issue)[
                0
            ].tolist()  # Note: Has to be row index not pandas index!
            issue_metric = self._issue_df[self._issue_df["issue"] == issue].iloc[0][
                "issue_metric"
            ]
            issue_explanation = (
                f"{issue_metric:.2f} -> "
                + self._issue_df[self._issue_df["issue"] == issue].iloc[0][
                    "issue_explanation"
                ]
            )

            data_issue = DataIssue(
                severity="warning", description=issue_explanation, rows=issue_rows
            )
            data_issues.append(data_issue)
            data_issue_severity.append(issue_metric)

        data_issue_order = np.argsort(data_issue_severity)
        if self._metric_mode == "min":
            data_issue_order = data_issue_order[::-1]

        spotlight.show(
            df,
            dtype=spotlight_dtype,
            issues=np.array(data_issues)[data_issue_order].tolist(),
        )
