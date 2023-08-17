# Supress numba deprecation warnings until umap fixes this
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

# Real imports
from uuid import uuid4
from typing import List, Literal, Dict, Callable, Optional

import pandas as pd
import numpy as np
import plotly.express as px

from renumics import spotlight
from renumics.spotlight.analysis.typing import DataIssue
from renumics.spotlight import Embedding
from renumics.spotlight import layout

from .utils import infer_feature_types, encode_normalize_features
from .detection import generate_metric_frames, detect_issues
from .explanation import explain_clusters
from .modeling import fit_outlier_detection_model
from .report import prepare_report


class SliceGuard:
    """
    The main class for detecting issues in your data
    """

    def show(
        self,
        data: pd.DataFrame,
        features: List[str],
        y: str = None,
        y_pred: str = None,
        metric: Callable = None,
        metric_mode: Literal["min", "max"] = None,
        drop_reference: Literal["overall", "parent"] = "overall",
        remove_outliers: bool = False,
        feature_types: Dict[
            str, Literal["raw", "nominal", "ordinal", "numerical", "embedding"]
        ] = {},
        feature_orders: Dict[str, list] = {},
        precomputed_embeddings: Dict[str, np.array] = {},
        embedding_models: Dict[str, str] = {},
        hf_auth_token=None,
        hf_num_proc=None,
        hf_batch_size=1,
    ):
        """
        Function to generate an interactive report that allows limited interactive exploration and
        serves as starting point for detailed analysis in spotlight.
        """
        df = data.copy()  # assign to shorter name

        print("Please wait. sliceguard is preparing your data.")
        (
            feature_types,
            encoded_data,
            mfs,
            clustering_df,
            clustering_cols,
            clustering_metric_cols,
            prereduced_embeddings,
            raw_embeddings,
        ) = self._prepare_data(
            df,
            features,
            y,
            y_pred,
            metric,
            remove_outliers,
            feature_types=feature_types,
            feature_orders=feature_orders,
            precomputed_embeddings=precomputed_embeddings,
            embedding_models=embedding_models,
            hf_auth_token=hf_auth_token,
            hf_num_proc=hf_num_proc,
            hf_batch_size=hf_batch_size,
        )

        if y is None and y_pred is None and metric_mode is None:
            metric_mode = "min"
            print(
                f"For outlier detection mode metric_mode will be set to {metric_mode} if not specified otherwise."
            )
        elif metric_mode is None:
            metric_mode = "max"
            print(
                f"You didn't specify metric_mode parameter. Using {metric_mode} as default."
            )

        prepare_report(mfs, clustering_df, clustering_cols, metric_mode, drop_reference)

    # TODO: Introduce control features to account for expected variations
    def find_issues(
        self,
        data: pd.DataFrame,
        features: List[str],
        y: str = None,
        y_pred: str = None,
        metric: Callable = None,
        min_support: int = None,
        min_drop: float = None,
        metric_mode: Literal["min", "max"] = None,
        drop_reference: Literal["overall", "parent"] = "overall",
        remove_outliers: bool = False,
        feature_types: Dict[
            str, Literal["raw", "nominal", "ordinal", "numerical", "embedding"]
        ] = {},
        feature_orders: Dict[str, list] = {},
        precomputed_embeddings: Dict[str, np.array] = {},
        embedding_models: Dict[str, str] = {},
        hf_auth_token=None,
        hf_num_proc=None,
        hf_batch_size=1,
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
        :param drop_reference: Determines what is the reference value for the drop. Overall is the metric on the whole dataset, parent is the parent cluster.
        :param remove_outliers: Account for outliers that disturb cluster detection.
        :param feature_types: Specify how your feature should be treated in encoding and normalizing.
        :param feature_orders: If your feature is ordinal, specify the order of that should be used for encoding. This is required for EVERY ordinal feature.
        :param precomputed_embeddings: Supply precomputed embeddings for raw columns. E.g. if repeatedly running checks on your data.
        :param embedding_model: Supply embedding models that should be used to compute embedding vectors from raw data.
        :param hf_auth_token: The authentification token used to download embedding models from the huggingface hub.
        :param hf_num_proc: Multiprocessing used in audio/image preprocessing.
        :param hf_batch_size: Batch size used in computing embeddings.
        """
        self._df = data  # safe that here to not modify original dataframe accidentally
        df = data.copy()  # assign to shorter name

        (
            feature_types,
            encoded_data,
            mfs,
            clustering_df,
            clustering_cols,
            clustering_metric_cols,
            prereduced_embeddings,
            raw_embeddings,
        ) = self._prepare_data(
            df,
            features,
            y,
            y_pred,
            metric,
            remove_outliers,
            feature_types=feature_types,
            feature_orders=feature_orders,
            precomputed_embeddings=precomputed_embeddings,
            embedding_models=embedding_models,
            hf_auth_token=hf_auth_token,
            hf_num_proc=hf_num_proc,
            hf_batch_size=hf_batch_size,
        )

        if len(mfs) > 0:
            overall_metric = mfs[0].overall.values[0]
            print(f"The overall metric value is {overall_metric}")

        if y is None and y_pred is None and metric_mode is None:
            metric_mode = "min"
            print(
                f"For outlier detection mode metric_mode will be set to {metric_mode} if not specified otherwise."
            )
        elif metric_mode is None:
            metric_mode = "max"
            print(
                f"You didn't specify metric_mode parameter. Using {metric_mode} as default."
            )

        group_dfs = detect_issues(
            mfs,
            clustering_df,
            clustering_cols,
            min_drop,
            min_support,
            metric_mode,
            drop_reference,
            remove_outliers,
        )

        num_issues = np.sum([group_df["issue"].sum() for group_df in group_dfs])

        print(f"Identified {num_issues} problematic slices.")

        # Construct the issue dataframe that is returned by this method
        issues = []

        issue_index = 0
        for hierarchy_level, (
            group_df,
            clustering_col,
            clustering_metric_col,
        ) in enumerate(zip(group_dfs, clustering_cols, clustering_metric_cols)):
            hierarchy_issues = group_df[group_df["issue"] == True].index
            for issue in hierarchy_issues:
                current_issue = {"id": issue_index, "level": hierarchy_level}
                issue_indices = clustering_df[
                    clustering_df[clustering_col] == issue
                ].index.values
                current_issue["indices"] = issue_indices

                issue_metric = clustering_df[clustering_df[clustering_col] == issue][
                    clustering_metric_col
                ].values[0]
                current_issue["metric"] = issue_metric

                issues.append(current_issue)

                issue_index += 1

        # Derive rules that are characteristic for each identified problem slice
        # This is done to help understanding of the problem reason
        # First stage this will be only importance values!
        issues = explain_clusters(
            features, feature_types, issues, df, prereduced_embeddings
        )

        self._issues = issues
        self._clustering_df = clustering_df
        self._clustering_cols = clustering_cols
        self._metric_mode = metric_mode
        self.embeddings = raw_embeddings

        return issues

    def report(
        self,
        spotlight_dtype: Dict[str, spotlight.dtypes.base.DType] = {},
        issue_portion: Optional[int | float] = None,
        non_issue_portion: Optional[int | float] = None,
    ):
        """
        Create an interactive report on the found issues in spotlight.
        :param spotlight_dtype: Define a datatype mapping for the interactive spotlight report. Will be passed to dtypes parameter of spotlight.show.
        :param issue_portion: The absolute or relative value of samples belonging to an issue that are shown in the report (for downsampling).
        :param non_issue_portion: The absolute or relative value of samples not belonging to an issue that are shown in the report (for downsampling).
        """
        # Some basic checks
        assert self._issues is not None

        df = self._df.copy()

        # Determine indices for downsampling if supplied.
        issue_indices = []
        for issue in self._issues:
            issue_indices.extend(issue["indices"])
        issue_indices = list(set(issue_indices))

        non_issue_indices = np.setdiff1d(df.index, issue_indices)

        if issue_portion is not None:
            if isinstance(issue_portion, float):
                issue_portion = round(min(1.0, issue_portion) * len(issue_indices))
            elif isinstance(issue_portion, int):
                pass  # Do nothing
            else:
                raise RuntimeError(
                    "Invalid value supplied to issue_portion. Must be int or float."
                )
            selected_issue_indices = np.random.choice(
                issue_indices, size=issue_portion, replace=False
            )
        else:
            selected_issue_indices = issue_indices

        if non_issue_portion is not None:
            if isinstance(non_issue_portion, float):
                non_issue_portion = round(
                    min(1.0, non_issue_portion) * len(issue_indices)
                )
            elif isinstance(non_issue_portion, int):
                pass  # Do nothing
            else:
                raise RuntimeError(
                    "Invalid value supplied to non_issue_portion. Must be int or float."
                )
            selected_non_issue_indices = np.random.choice(
                issue_indices, size=non_issue_portion, replace=False
            )
        else:
            selected_non_issue_indices = non_issue_indices

        # Downsample the dataframe
        selected_dataframe_indices = np.concatenate(
            (selected_issue_indices, selected_non_issue_indices)
        )
        selected_dataframe_rows = np.where(df.index.isin(selected_dataframe_indices))[
            0
        ].tolist()
        df = df.iloc[selected_dataframe_rows]

        # Insert embeddings if they were computed
        embedding_dtypes = {}
        for embedding_col, embeddings in self.embeddings.items():
            report_col_name = f"sg_emb_{embedding_col}"
            df[report_col_name] = np.array([e.tolist() for e in embeddings])[
                selected_dataframe_rows
            ].tolist()
            embedding_dtypes[report_col_name] = Embedding

        data_issue_severity = []
        data_issues = []
        for issue in self._issues:
            # Note: Has to be row index not pandas index! Also note that the expression should be enough to filter out items that are not in the dataframe
            # because of downsampling. However, take care when changing something.
            issue_rows = np.where(df.index.isin(issue["indices"]))[0].tolist()
            issue_metric = issue["metric"]
            issue_title = f"{issue_metric:.2f} -> " + issue["explanation"]
            predicate_strings = [
                "{1:.1f} < {0} < {2:.1f}".format(
                    x["column"], x["minimum"], x["maximum"]
                )
                for x in issue["predicates"]
                if ("minimum" in x and "maximum" in x)
            ]
            issue_explanation = "; ".join(predicate_strings)

            data_issue = DataIssue(
                severity="medium",
                title=issue_title,
                description=issue_explanation,
                rows=issue_rows,
                columns=[x["column"] for x in issue["predicates"]],
            )
            data_issues.append(data_issue)
            data_issue_severity.append(issue_metric)

        data_issue_order = np.argsort(data_issue_severity)
        if self._metric_mode == "min":
            data_issue_order = data_issue_order[::-1]

        if hasattr(self, "_generated_y_pred"):
            df["sg_y_pred"] = self._generated_y_pred[selected_dataframe_rows]

        issue_list = np.array(data_issues)[data_issue_order].tolist()

        spotlight.show(
            df,
            dtype={**spotlight_dtype, **embedding_dtypes},
            issues=issue_list,
            layout=layout.layout(
                [
                    [layout.table()],
                    [layout.similaritymap()],
                    [layout.histogram()],
                ],
                [[layout.widgets.Inspector()], [layout.widgets.Issues()]],
            ),
        )
        return (
            df,
            issue_list,
        )  # Return the create report dataframe in case caller wants to process it

    def _prepare_data(
        self,
        data: pd.DataFrame,
        features: List[str],
        y: str,
        y_pred: str,
        metric: Callable,
        remove_outliers: bool = False,
        feature_types: Dict[
            str, Literal["raw", "nominal", "ordinal", "numerical", "embedding"]
        ] = {},
        feature_orders: Dict[str, list] = {},
        precomputed_embeddings: Dict[str, np.array] = {},
        embedding_models: Dict[str, str] = {},
        hf_auth_token=None,
        hf_num_proc=None,
        hf_batch_size=1,
    ):
        assert (
            all([(f in data.columns or f in precomputed_embeddings) for f in features])
        ) and (
            ((y_pred is not None) and (y is not None))  # Completly supervised case
            or ((y_pred is None) and (y is None))
        )  # Completely unsupervised case (outlier based)  # check presence of all columns

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
            hf_num_proc,
            hf_batch_size,
            df,
        )

        # If y and y_pred are non use an outlier detection algorithm to detect potential issues in the data.
        # Lateron there could be also a case where y is given but no y_pred is given. Then just train a task specific surrogate model.
        # However, besides regression and classification cases this could be much work. Consider using FLAML or other automl tooling here.
        if y is None and y_pred is None:
            print(
                "You didn't supply ground-truth labels and predictions. Will fit outlier detection model to find anomal slices instead."
            )
            ol_scores = fit_outlier_detection_model(encoded_data)
            ol_model_id = str(uuid4())

            y = f"{ol_model_id}_y"
            df[y] = ol_scores

            y_pred = f"{ol_model_id}_y_pred"
            df[y_pred] = ol_scores

            def return_y_pred_mean(y, y_pred):
                return np.mean(y_pred)

            metric = return_y_pred_mean

            self._generated_y_pred = ol_scores

        # Perform detection of problematic clusters based on the given features
        # 1. A hierarchical clustering is performed and metrics are calculated for all hierarchies
        # 2. hierarchy level that is most indicative of a real problem is then determined
        # 3. the reason for the problem e.g. feature combination or rule that is characteristic for the cluster is determined.

        (
            mfs,
            clustering_df,
            clustering_cols,
            clustering_metric_cols,
        ) = generate_metric_frames(
            encoded_data, df, y, y_pred, metric, feature_types, remove_outliers
        )

        return (
            feature_types,
            encoded_data,
            mfs,
            clustering_df,
            clustering_cols,
            clustering_metric_cols,
            prereduced_embeddings,
            raw_embeddings,
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
