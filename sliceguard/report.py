import pandas as pd
import numpy as np
from typing import List, Literal, Dict

import plotly.express as px
from dash import Dash, html, dcc, callback, Output, Input


def prepare_report(mfs, clustering_df, clustering_cols, metric_mode):
    all_drops = []
    all_supports = []
    for mf, clustering_col in zip(mfs, clustering_cols):
        # Calculate cluster support
        drops = (
            mf.overall.values[0] - mf.by_group.values[:, 0]
            if metric_mode == "max"
            else mf.by_group.values[:, 0] - mf.overall.values[0]
        )

        supports = [
            (clustering_df[clustering_col] == cluster).sum()
            for cluster in mf.by_group.index
        ]
        all_drops.extend(drops)
        all_supports.extend(supports)

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

    app.layout = html.Div([
    html.H1(children='sliceguard Interactive Report', style={'textAlign':'center'}),
    dcc.Graph(figure=fig)
    ])

    app.run()
