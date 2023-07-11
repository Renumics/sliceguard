# Supress numba deprecation warnings until umap fixes this
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

# Real imports
from typing import List, Literal, Dict, Callable

import pandas as pd
import numpy as np
import plotly.express as px

from renumics import spotlight
from renumics.spotlight.analysis.typing import DataIssue
from renumics.spotlight import layout

from .utils import infer_feature_types, encode_normalize_features
from .detection import generate_metric_frames, detect_issues
from .explanation import explain_clusters


class SliceGuard:
    """
    The main class for detecting issues in your data
    """

    # TODO: Introduce control features to account for expected variations
    def find_issues(
        self,
        data: pd.DataFrame,
        features: List[str],
        y: str,
        y_pred: str,
        metric: Callable,
        min_support: int = None,
        min_drop: float = None,
        metric_mode: Literal["min", "max"] = "max",
        feature_types: Dict[
            str, Literal["raw", "nominal", "ordinal", "numerical", "embedding"]
        ] = {},
        feature_orders: Dict[str, list] = {},
        precomputed_embeddings: Dict[str, np.array] = {},
        embedding_models: Dict[str, str] = {},
        hf_auth_token=None,
    ):
        """
        Find slices that are classified badly by your model.

        :param data: A pandas dataframe containing your data.
        :param features: A list of columns that contains features to feed into your model but also metadata.
        :param y: The column containing the ground-truth label.
        :param y_pred: The column containing your models prediction.
        :param metric: A callable metric function that must correspond to the form metric(y_true, y_pred) -> scikit-learn style.
        :min_support: Minimum support for clusters that are listed as issues. If you are more looking towards outliers choose small values, if you target biases choose higher values.
        :min_drop: Minimum metric drop a cluster has to have to be counted as issue compared to the result on the whole dataset.
        :param metric_mode: What do you optimize your metric for? max is the right choice for accuracy while e.g. min is good for regression error.
        :param feature_types: Specify how your feature should be treated in encoding and normalizing.
        :param feature_orders: If your feature is ordinal, specify the order of that should be used for encoding. This is required for EVERY ordinal feature.
        :param precomputed_embeddings: Supply precomputed embeddings for raw columns. E.g. if repeatedly running checks on your data.
        :param embedding_model: Supply embedding models that should be used to compute embedding vectors from raw data.
        :param hf_auth_token: The authentification token used to download embedding models from the huggingface hub.
        """

        assert (
            (
                all(
                    [
                        (f in data.columns or f in precomputed_embeddings)
                        for f in features
                    ]
                )
            )
            and (y in data.columns)
            and (y_pred in data.columns)
        )  # check presence of all columns
        df = data  # just rename the variable for shorter naming

        # Try to infer the column dtypes
        feature_types = infer_feature_types(
            features, feature_types, precomputed_embeddings, df
        )

        # TODO: Potentially also explicitely check for univariate and bivariate fairness issues, however start with the more generic variant
        # See also connection with full report functionality. It makes sense to habe a feature and a samples based view!

        # Encode the features for clustering according to inferred types
        encoded_data, prereduced_embeddings, raw_embeddings = encode_normalize_features(
            features,
            feature_types,
            feature_orders,
            precomputed_embeddings,
            embedding_models,
            hf_auth_token,
            df,
        )

        # Perform detection of problematic clusters based on the given features
        # 1. A hierarchical clustering is performed and metrics are calculated for all hierarchies
        # 2. hierarchy level that is most indicative of a real problem is then determined
        # 3. the reason for the problem e.g. feature combination or rule that is characteristic for the cluster is determined.

        (
            mfs,
            clustering_df,
            clustering_cols,
            clustering_metric_cols,
        ) = generate_metric_frames(encoded_data, df, y, y_pred, metric)

        if len(mfs) > 0:
            overall_metric = mfs[0].overall.values[0]
            print(f"The overall metric value is {overall_metric}")

        group_dfs = detect_issues(
            mfs, clustering_df, clustering_cols, min_drop, min_support, metric_mode
        )

        num_issues = np.sum([group_df["issue"].sum() for group_df in group_dfs])

        print(f"Identified {num_issues} problematic slices.")

        # Construct the issue dataframe that is returned by this method
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

        # Derive rules that are characteristic for each identified problem slice
        # This is done to help understanding of the problem reason
        # First stage this will be only importance values!
        issue_df = explain_clusters(
            features, feature_types, issue_df, df, prereduced_embeddings
        )

        self._issue_df = issue_df
        self._clustering_df = clustering_df
        self._clustering_cols = clustering_cols
        self._metric_mode = metric_mode
        self._df = df
        self.embeddings = raw_embeddings

        return issue_df

    def report(self, spotlight_dtype: Dict[str, spotlight.dtypes.base.DType] = {}):
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
                severity="medium",
                title=issue_explanation,
                description=issue_explanation,
                rows=issue_rows,
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
            layout=layout.layout(
                [
                    [layout.table()],
                    [layout.similaritymap()],
                    [layout.histogram()],
                ],
                [[layout.widgets.Inspector()], [layout.widgets.Issues()]],
            ),
        )

    def _plot_clustering_overview(self):
        """
        Debugging method to get an overview on the last clustering structure.
        """
        # for i in range(len(self._clustering_cols)):
        # cur_clustering_cols = self._clustering_cols[:i+1]
        cur_clustering_cols = self._clustering_cols
        plotting_df = pd.concat((self._clustering_df, self._issue_df), axis=1)
        plotting_df["is_issue"] = plotting_df["issue"] != -1
        plotting_df["issue_metric_str"] = plotting_df["issue_metric"].apply(
            lambda x: "{0:.2f}".format(x)
            if (isinstance(x, float) or isinstance(x, np.floating)) and not np.isnan(x)
            else ""
        )
        fig = px.treemap(
            plotting_df,
            path=cur_clustering_cols,
            color="is_issue",
            color_discrete_map={"(?)": "lightgrey", True: "gold", False: "darkblue"},
            custom_data="issue_metric_str",
        )
        fig.data[0].texttemplate = "%{customdata[0]}"
        fig.data[0].textinfo = "none"
        # fig.write_image(f"clustering_{i}.png", scale=4)

        fig.show()
