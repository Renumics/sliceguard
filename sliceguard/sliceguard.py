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
from .modeling import fit_outlier_detection_model, fit_classification_regression_model
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
        embedding_weights: Dict[str, float] = {},
        hf_auth_token=None,
        hf_num_proc=None,
        hf_batch_size=1,
        automl_task="classification",
        automl_split_key=None,
        automl_train_split=None,
        automl_time_budget=20.0,
        automl_use_full_embeddings=False,
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
            projection,
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
            embedding_weights=embedding_weights,
            hf_auth_token=hf_auth_token,
            hf_num_proc=hf_num_proc,
            hf_batch_size=hf_batch_size,
            automl_split_key=automl_split_key,
            automl_train_split=automl_train_split,
            automl_task=automl_task,
            automl_time_budget=automl_time_budget,
            automl_use_full_embeddings=automl_use_full_embeddings,
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
        n_slices: int = None,
        criterion: Literal["drop", "support", "drop*support"] = None,
        metric_mode: Literal["min", "max"] = None,
        drop_reference: Literal["overall", "parent"] = "overall",
        remove_outliers: bool = False,
        feature_types: Dict[
            str, Literal["raw", "nominal", "ordinal", "numerical", "embedding"]
        ] = {},
        feature_orders: Dict[str, list] = {},
        precomputed_embeddings: Dict[str, np.array] = {},
        embedding_models: Dict[str, str] = {},
        embedding_weights: Dict[str, float] = {},
        hf_auth_token=None,
        hf_num_proc=None,
        hf_batch_size=1,
        automl_task="classification",
        automl_split_key=None,
        automl_train_split=None,
        automl_time_budget=20.0,
        automl_use_full_embeddings=False,
    ):
        """
        Find slices that are classified badly by your model.

        :param data: A pandas dataframe containing your data.
        :param features: A list of columns that contains features to feed into your model but also metadata.
        :param y: The column containing the ground-truth label.
        :param y_pred: The column containing your models prediction.
        :param metric: A callable metric function that must correspond to the form metric(y_true, y_pred) -> scikit-learn style.
        :param min_support: Minimum support for clusters that are listed as issues. If you are more looking towards outliers choose small values, if you target biases choose higher values.
        :param min_drop: Minimum metric drop a cluster has to have to be counted as issue compared to the result on the whole dataset.
        :param n_slices: Number of slices to return for review. Alternative interface to min_drop and min_support.
        :param criterion: Criterion after which the slices get sorted when using n_slices. One of drop, support or drop*support.
        :param metric_mode: What do you optimize your metric for? max is the right choice for accuracy while e.g. min is good for regression error.
        :param drop_reference: Determines what is the reference value for the drop. Overall is the metric on the whole dataset, parent is the parent cluster.
        :param remove_outliers: Account for outliers that disturb cluster detection.
        :param feature_types: Specify how your feature should be treated in encoding and normalizing.
        :param feature_orders: If your feature is ordinal, specify the order of that should be used for encoding. This is required for EVERY ordinal feature.
        :param precomputed_embeddings: Supply precomputed embeddings for raw columns. E.g. if repeatedly running checks on your data.
        :param embedding_models: Supply embedding models that should be used to compute embedding vectors from raw data.
        :param embedding_weight: Lower the influence of embedding values in mixed inferences by setting it lower than 1.0.
        :param hf_auth_token: The authentification token used to download embedding models from the huggingface hub.
        :param hf_num_proc: Multiprocessing used in audio/image preprocessing.
        :param hf_batch_size: Batch size used in computing embeddings.
        :param automl_task: The task specification for training an own model. Has to be one of classification or regression.
        :param automl_split_key: Column used for splitting the data.
        :param automl_train_split: The value used for marking the train split. If supplied, rest of data will be used as validation set. If not supplied using crossvalidation.
        :param automl_time_budget: The time budget used for training an own model.
        :param automl_use_full_embeddings: Wether to use the raw embeddings instead of the pre-reduced ones when training a model.
        """

        # Validate if there is invalid configuration of slice return config
        if (
            min_drop is None
            and min_support is None
            and n_slices is None
            and criterion is None
        ):
            n_slices = 20
            criterion = "drop"
        else:
            if not (
                (
                    min_drop is not None
                    and min_support is not None
                    and n_slices is None
                    and criterion is None
                )
                or (
                    min_drop is None
                    and min_support is None
                    and n_slices is not None
                    and criterion is not None
                )
            ):
                raise RuntimeError(
                    "Invalid Configuration: Use either n_slices and criterion or min_drop and min_support!"
                )

        self._df = data  # safe that here to not modify original dataframe accidentally
        df = data.copy()  # assign to shorter name

        (
            feature_types,
            encoded_data,
            projection,
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
            embedding_weights=embedding_weights,
            hf_auth_token=hf_auth_token,
            hf_num_proc=hf_num_proc,
            hf_batch_size=hf_batch_size,
            automl_split_key=automl_split_key,
            automl_train_split=automl_train_split,
            automl_task=automl_task,
            automl_time_budget=automl_time_budget,
            automl_use_full_embeddings=automl_use_full_embeddings,
        )

        if len(mfs) > 0:
            overall_metric = mfs[0].overall.values[0]
            print(f"The overall metric value is {overall_metric}")

        if y is None and y_pred is None and metric_mode is None:
            metric_mode = "min"
            print(
                f"For outlier detection mode metric_mode will be set to {metric_mode} if not specified otherwise."
            )

        elif y_pred is None and metric_mode is None:
            if automl_task == "classification":
                metric_mode = "max"
            elif automl_task == "regression":
                metric_mode = "min"
            print(
                f"You didn't specify metric_mode. For task {automl_task} using {metric_mode} as a default."
            )
        elif metric_mode is None:
            metric_mode = "max"
            print(
                f"You didn't specify metric_mode parameter. Using {metric_mode} as default."
            )

        assert metric_mode is not None

        group_dfs = detect_issues(
            mfs,
            clustering_df,
            clustering_cols,
            min_drop,
            min_support,
            n_slices,
            criterion,
            metric_mode,
            drop_reference,
            remove_outliers,
        )

        num_issues = np.sum([group_df["issue"].sum() for group_df in group_dfs])

        print(f"Identified {num_issues} problematic slices.")

        # Construct the issue list that is returned by this method
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

                issue_rows = np.where(clustering_df[clustering_col] == issue)[0]
                current_issue["rows"] = issue_rows

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
        self._projection = projection
        self.embeddings = raw_embeddings

        return issues

    def report(
        self,
        spotlight_dtype: Dict[str, spotlight.dtypes.base.DType] = {},
        issue_portion: Optional[int | float] = None,
        non_issue_portion: Optional[int | float] = None,
        host: str = "127.0.0.1",
        port: int = "auto",
        no_browser: bool = False,
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
        issue_rows = []
        for issue in self._issues:
            issue_rows.extend(issue["rows"])
        issue_rows = list(set(issue_rows))

        non_issue_rows = np.setdiff1d(np.arange(len(df)), issue_rows)

        if issue_portion is not None:
            if isinstance(issue_portion, float):
                issue_portion = round(min(1.0, issue_portion) * len(issue_rows))
            elif isinstance(issue_portion, int):
                pass  # Do nothing
            else:
                raise RuntimeError(
                    "Invalid value supplied to issue_portion. Must be int or float."
                )
            selected_issue_rows = np.random.choice(
                issue_rows, size=issue_portion, replace=False
            )
        else:
            selected_issue_rows = issue_rows

        if non_issue_portion is not None:
            if isinstance(non_issue_portion, float):
                non_issue_portion = round(
                    min(1.0, non_issue_portion) * len(non_issue_rows)
                )
            elif isinstance(non_issue_portion, int):
                pass  # Do nothing
            else:
                raise RuntimeError(
                    "Invalid value supplied to non_issue_portion. Must be int or float."
                )
            selected_non_issue_rows = np.random.choice(
                non_issue_rows, size=non_issue_portion, replace=False
            )
        else:
            selected_non_issue_rows = non_issue_rows

        # Downsample the dataframe
        selected_dataframe_rows = np.sort(
            np.concatenate(
                (selected_issue_rows, selected_non_issue_rows)
            )  # Do not change the order of the dataframe here. This was a hard to find bug!!!
        )

        df = df.iloc[selected_dataframe_rows]

        # Insert the computed data projection if it exists.
        # Could not be the case when dealing with one or two categorical variables as there is no hnne projection involved for computing metrics.
        # Also if hnne fails and hdbscan is used as fallback projection will be None.
        embedding_dtypes = {}

        if self._projection is not None:
            df["sg_projection"] = self._projection[selected_dataframe_rows].tolist()
            embedding_dtypes["sg_projection"] = Embedding

        # Insert embeddings if they were computed
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
            issue_metric = issue["metric"]
            issue_title = f"Metric: {issue_metric:.2f} | Cause: {issue['explanation'][0]['column']}"

            issue_explanation = ""

            num_features_explanation = 3

            importance_strings = [
                f"{x['column']}, ({x['importance']:.2f})"
                for x in issue["explanation"][:num_features_explanation]
                if ("column" in x and "importance" in x)
            ]
            if len(importance_strings) > 0:
                issue_explanation += "Feature Importances: " + "; ".join(
                    importance_strings
                )

            predicate_strings = [
                f"{x['minimum']:.1f}  < {x['column']} < {x['maximum']:.1f}"
                for x in issue["explanation"][:num_features_explanation]
                if ("minimum" in x and "maximum" in x)
            ]

            if len(predicate_strings) > 0:
                issue_explanation += "\n\nFeature Ranges: " + "; ".join(
                    predicate_strings
                )

            issue_rows = np.where(np.isin(selected_dataframe_rows, issue["rows"]))[
                0
            ].tolist()

            data_issue = DataIssue(
                severity="medium",
                title=issue_title,
                description=issue_explanation,
                rows=issue_rows,
                columns=[
                    x["column"] for x in issue["explanation"][:num_features_explanation]
                ],
            )
            data_issues.append(data_issue)
            data_issue_severity.append(issue_metric)

        data_issue_order = np.argsort(data_issue_severity)
        if self._metric_mode == "min":
            data_issue_order = data_issue_order[::-1]

        if hasattr(self, "_generated_y_pred"):
            df["sg_y_pred"] = self._generated_y_pred[selected_dataframe_rows]

        issue_list = np.array(data_issues)[data_issue_order].tolist()

        if not no_browser:
            spotlight.show(
                df,
                dtype={**spotlight_dtype, **embedding_dtypes},
                host=host,
                port=port,
                issues=issue_list,
                layout=layout.layout(
                    [
                        [layout.table()],
                        [
                            layout.similaritymap(
                                columns=["sg_projection"]
                                if self._projection is not None
                                else None
                            )
                        ],
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
        embedding_weights: Dict[str, float] = {},
        hf_auth_token=None,
        hf_num_proc=None,
        hf_batch_size=1,
        automl_task="classification",
        automl_split_key=None,
        automl_train_split=None,
        automl_time_budget=None,
        automl_use_full_embeddings=False,
    ):
        assert (
            all([(f in data.columns or f in precomputed_embeddings) for f in features])
        ) and (
            (
                (y_pred is not None) and (y is not None) and (metric is not None)
            )  # Completly supervised case
            or (
                (y_pred is None) and (y is not None) and (metric is not None)
            )  # fit own model
            or (
                (y_pred is None) and (y is None) and (metric is None)
            )  # Completely unsupervised case (outlier based)
        )

        df = data  # just rename the variable for shorter naming

        # Try to infer the column dtypes
        feature_types = infer_feature_types(
            features, feature_types, precomputed_embeddings, df
        )

        # TODO: Potentially also explicitely check for univariate and bivariate fairness issues, however start with the more generic variant
        # See also connection with full report functionality. It makes sense to habe a feature and a samples based view!

        run_mode = None

        if y is None and y_pred is None:
            mode = "outlier"
        elif y_pred is None:
            mode = "automl"
        elif y is not None and y_pred is not None:
            mode = "native"
        else:
            raise RuntimeError("Could not determine run mode.")

        # Encode the features for clustering according to inferred types
        encoded_data, prereduced_embeddings, raw_embeddings = encode_normalize_features(
            features,
            feature_types,
            feature_orders,
            precomputed_embeddings,
            embedding_models,
            embedding_weights,
            hf_auth_token,
            hf_num_proc,
            hf_batch_size,
            df,
            mode,
        )

        # If y and y_pred are non use an outlier detection algorithm to detect potential issues in the data.
        # If y is given but no y_pred is given just train a task specific surrogate model.
        # Currently only classification and regression are supported.
        if mode == "outlier":
            print(
                "You didn't supply ground-truth labels and predictions. Will fit outlier detection model to find anomal slices instead."
            )
            ol_scores = fit_outlier_detection_model(
                np.concatenate((encoded_data, raw_embeddings), axis=1)
                if automl_use_full_embeddings
                else encoded_data,
            )
            ol_model_id = str(uuid4())

            y = f"{ol_model_id}_y"
            df[y] = ol_scores

            y_pred = f"{ol_model_id}_y_pred"
            df[y_pred] = ol_scores

            def return_y_pred_mean(y, y_pred):
                return np.mean(y_pred)

            metric = return_y_pred_mean

            self._generated_y_pred = ol_scores

        elif mode == "automl":
            y_pred = "sg_y_pred"

            X_data = [encoded_data]
            if automl_use_full_embeddings:
                print(
                    f"Using {len(list(raw_embeddings.values()))} raw embeddings in fitting. Consider increasing the time budget."
                )
                for v in raw_embeddings.values():
                    X_data.append(v)

            y_preds = fit_classification_regression_model(
                encoded_data=np.concatenate(X_data, axis=1)
                if automl_use_full_embeddings
                else encoded_data,
                ys=df[y].values,
                task=automl_task,
                split=df[automl_split_key].values
                if automl_split_key is not None
                else None,
                train_split=automl_train_split,
                time_budget=automl_time_budget,
            )

            df[y_pred] = y_preds

            self._generated_y_pred = df[y_pred].values

        # Perform detection of problematic clusters based on the given features
        # 1. A hierarchical clustering is performed and metrics are calculated for all hierarchies
        # 2. hierarchy level that is most indicative of a real problem is then determined
        # 3. the reason for the problem e.g. feature combination or rule that is characteristic for the cluster is determined.

        (
            projection,
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
            projection,
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
