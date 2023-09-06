# Supress numba deprecation warnings until umap fixes this
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

# Real imports
from uuid import uuid4
from typing import List, Literal, Dict, Callable, Optional, Tuple, Union

import pandas as pd
import numpy as np


from renumics import spotlight
from renumics.spotlight.analysis.typing import DataIssue
from renumics.spotlight import Embedding
from renumics.spotlight import layout

from .utils import infer_feature_types, encode_normalize_features
from .detection import generate_metric_frames, detect_issues
from .explanation import explain_clusters
from .modeling import fit_outlier_detection_model, fit_classification_regression_model


class SliceGuard:
    """
    The main class for detecting issues in your data
    """

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
    ) -> List[dict]:
        """
        Find slices that are classified badly by your model.

        :param data: A pandas dataframe containing your data.
        :param features: A list of feature column names sliceguard should use for for identifying problematic data clusters.
        :param y: Name of the dataframe column containing your ground-truth labels.
        :param y_pred: Name of the dataframe column containing your model's predictions.
        :param metric: A callable metric function that must correspond to the form metric(y_true, y_pred) -> scikit-learn style.
        :param min_support: Minimum support for a cluster to be listed as an issue.
        :param min_drop: Minimum metric drop for a cluster to be listed as an issue.
        :param n_slices: Number of problematic clusters find_issues should return after sorting them by a criterion specified by the "criterion" parameter.
        :param criterion: Criterion after which the slices get sorted when using n_slices. One of drop, support or drop*support.
        :param metric_mode: Optimization goal for your metric. max is the right choice for accuracy while e.g. min is good for regression error.
        :param drop_reference: Reference value for calculating the drop for a cluster. Default is "overall" which calculates the difference to the overall metric.
                "parent" will calculate the drop relative to each clusters parent cluster. Use the second option for getting more diverse results,
                e.g., getting problematic clusters in each class when dealing with image classification instead of focussing on the most difficult class.
        :param remove_outliers: Filter metric outliers in identified clusters. Especially useful if metric is unbounded and can heavily
            distort a clusters overall metric. Will be significantly more computationally expensive.
        :param feature_types: Specify the types of your features if sliceguard doesn't detect them properly. Can be "nominal", "ordinal", "numerical" for scalar values.
            Can be "raw" for filepaths to unstructured data. Can be "embedding" for embedding vectors.
        :param feature_orders: Specify the order of ordinal feature values that should be used for encoding. This is required for EVERY ordinal feature
            that is not specified by pandas categorical ordered datatypes.
        :param precomputed_embeddings: Supply precomputed embeddings for raw columns. Form should be precomputed_embeddings={"image": image_embeddings}.
            This is especially useful if you run repeated checks on your data and you want to compute embeddings only once.
        :param embedding_models: Supply huggingface model identifiers used for computing embeddings on specific columns.
            Form should be embedding_models={"image": "google/vit-base-patch16-224"}.
        :param embedding_weights: Specify how much each computed embedding is weighted in the cluster search. Useful to lower the influence of an embedding
            by setting the parameter lower than 1.0.
        :param hf_auth_token: The authentification token used to download embedding models from the huggingface hub.
        :param hf_num_proc: Number of processes used in embedding computation.
        :param hf_batch_size: Batch size used for embedding computation.
        :param automl_task: The task specification for training a model. Has to be one of classification or regression. Used when only supplying labels.
        :param automl_split_key: Name of column used for splitting the data when sliceguard trains a model.
        :param automl_train_split: The value used for marking the train split when sliceguard trains a model. If supplied, rest of data will be used as validation set.
            If not supplied using crossvalidation.
        :param automl_time_budget: The time budget used by sliceguard for training a model.
        :param automl_use_full_embeddings: Wether to use the raw embeddings instead of the pre-reduced ones when training a model. Can potentially improve performance.
        :rtype: List of issues, represented as python dicts.
        """

        # If nothing is given set a default config
        if (
            min_drop is None
            and min_support is None
            and n_slices is None
            and criterion is None
        ):
            n_slices = 20
            criterion = "drop"
        elif n_slices is not None and criterion is None:
            criterion = "drop"

        self._df = data  # safe that here to not modify original dataframe accidentally
        df = data.copy()  # assign to shorter name

        # also safe originally supplied parameters for y and y_pred here
        self._y = y
        self._y_pred = y_pred

        (
            feature_types,
            raw_feature_types,
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
        self._features = features
        self._feature_types = feature_types
        self._raw_feature_types = raw_feature_types
        self.embeddings = raw_embeddings

        return issues

    def report(
        self,
        spotlight_dtype: Dict[str, spotlight.dtypes.base.DType] = {},
        issue_portion: Optional[Union[int, float]] = None,
        non_issue_portion: Optional[Union[int, float]] = None,
        host: str = "127.0.0.1",
        port: int = "auto",
        no_browser: bool = False,
    ) -> Tuple[pd.DataFrame, List[DataIssue], Dict[str, spotlight.dtypes.base.DType]]:
        """
        Create an interactive report on the found issues in spotlight.

        :param spotlight_dtype: Define a datatype mapping for the interactive spotlight report. Will be passed to dtypes parameter of spotlight.show. Form is spotlight_dtype={"image": spotlight.Image}.
        :param issue_portion: The absolute or relative value of samples belonging to an issue that are shown in the report (for downsampling).
        :param non_issue_portion: The absolute or relative value of samples not belonging to an issue that are shown in the report (for downsampling).
        :param host: The host spotlight should be started on. Default is 127.0.0.1.
        :param port: The port spotlight should be started on. Default is "auto".
        :param no_browser: Do not start spotlight but just return the dataframe and issues. Useful for programmatic issue evaluation.
        :rtype: Tuple in the format (enriched dataframe, list of spotlight DataIssues, spotlight datatype mapping dict, spotlight layout).
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
            np.concatenate((selected_issue_rows, selected_non_issue_rows)).astype(
                int
            )  # Do not change the order of the dataframe here. This was a hard to find bug!!! Note that also astype(int) is important in case one of the arrays is empty. Else auto converted to float.
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

            range_predicate_strings = [
                f"{x['minimum']:.1f}  < {x['column']} < {x['maximum']:.1f}"
                for x in issue["explanation"][:num_features_explanation]
                if ("minimum" in x and "maximum" in x)
            ]

            mode_predicate_strings = [
                f"{x['column']} (mode) = {x['mode']}"
                for x in issue["explanation"][:num_features_explanation]
                if ("mode" in x)
            ]

            predicate_strings = range_predicate_strings + mode_predicate_strings

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

        if hasattr(self, "_generated_y_probs") and hasattr(self, "_classes"):
            for class_idx, label in enumerate(self._classes):
                df[f"sg_p_{label}"] = self._generated_y_probs[:, class_idx].tolist()

        spotlight_issue_list = np.array(data_issues)[data_issue_order].tolist()

        spotlight_dtypes = {**spotlight_dtype, **embedding_dtypes}

        spotlight_inspector_fields = []
        for col in self._features:
            if self._feature_types[col] == "raw":
                if self._raw_feature_types[col] == "image":
                    spotlight_inspector_fields.append(layout.lenses.image(col))
                elif self._raw_feature_types[col] == "audio":
                    spotlight_inspector_fields.append(layout.lenses.audio(col))
                    spotlight_inspector_fields.append(layout.lenses.spectrogram(col))
                elif self._raw_feature_types[col] == "text":
                    spotlight_inspector_fields.append(layout.lenses.text(col))
            if (
                self._feature_types[col] == "numerical"
                or self._feature_types[col] == "nominal"
                or self._feature_types[col] == "ordinal"
            ):
                spotlight_inspector_fields.append(layout.lenses.scalar(col))

        if self._y is not None:
            spotlight_inspector_fields.append(layout.lenses.scalar(self._y))

        if self._y_pred is not None:
            spotlight_inspector_fields.append(layout.lenses.scalar(self._y_pred))
        else:
            spotlight_inspector_fields.append(layout.lenses.scalar("sg_y_pred"))

        spotlight_layout = layout.layout(
            [
                [layout.table()],
                [
                    layout.similaritymap(
                        columns=["sg_projection"]
                        if self._projection is not None
                        else None
                    )
                ],
                [
                    layout.histogram(
                        "Histogram",
                        column=next(
                            (
                                col
                                for col in self._features
                                if self._feature_types[col]
                                in ["numerical", "nominal", "ordinal"]
                            ),
                            None,
                        ),
                    )
                ],
            ],
            [
                [layout.inspector("Inspector", spotlight_inspector_fields)],
                [layout.issues()],
            ],
        )

        if not no_browser:
            spotlight.show(
                df,
                dtype=spotlight_dtypes,
                host=host,
                port=port,
                issues=spotlight_issue_list,
                layout=spotlight_layout,
            )
        return (
            df,
            spotlight_issue_list,
            spotlight_dtypes,
            spotlight_layout,
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
        feature_types, feature_orders, raw_feature_types = infer_feature_types(
            features, feature_types, feature_orders, precomputed_embeddings, df
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
            raw_feature_types,
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

            y_preds, y_probs, classes = fit_classification_regression_model(
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

            if classes is not None:
                self._classes = classes
            if y_probs is not None:
                self._generated_y_probs = y_probs
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
            raw_feature_types,
            encoded_data,
            projection,
            mfs,
            clustering_df,
            clustering_cols,
            clustering_metric_cols,
            prereduced_embeddings,
            raw_embeddings,
        )
