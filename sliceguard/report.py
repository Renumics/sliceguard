import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame
from typing import List, Literal, Dict

import plotly.express as px
from dash import Dash, html, dcc, callback, Output, Input


def prepare_report(
    mfs: List[MetricFrame],
    clustering_df: pd.DataFrame,
    clustering_cols: List[str],
    metric_mode: Literal["min", "max"],
    drop_reference: Literal["overall", "parent"],
):
    all_drops = []
    all_supports = []
    previous_clustering_col = None
    for mf, clustering_col in zip(mfs, clustering_cols):
        # Calculate cluster support

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

        drops = (
            drop_reference_value - mf.by_group.values[:, 0]
            if metric_mode == "max"
            else mf.by_group.values[:, 0] - drop_reference_value
        )

        supports = [
            (clustering_df[clustering_col] == cluster).sum()
            for cluster in mf.by_group.index
        ]
        all_drops.extend(drops)
        all_supports.extend(supports)

        previous_clustering_col = clustering_col

    drop_support_df = pd.DataFrame(data={"support": all_supports, "drop": all_drops})
    fig = px.density_heatmap(
        drop_support_df,
        x="drop",
        y="support",
        nbinsx=10,
        nbinsy=10,
        color_continuous_scale="Viridis",
        text_auto=True,
    )

    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.H1(
                children="sliceguard Interactive Report", style={"textAlign": "center"}
            ),
            dcc.Graph(figure=fig),
        ]
    )

    app.run()
