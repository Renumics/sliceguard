import pandas as pd
import numpy as np
from jiwer import wer
from renumics import spotlight
from renumics.spotlight import Embedding
from renumics.spotlight.analysis.typing import DataIssue

from segment_guard import SegmentGuard


def wer_metric(y_true, y_pred):
    return np.mean([wer(s_y, s_pred) for s_y, s_pred in zip(y_true, y_pred)])


def test_segment_guard():
    df = pd.read_json("tests/predictions_embs.json")
    df = df[df["accent"] != ""]
    sg = SegmentGuard()
    issue_df = sg.find_issues(
        df,
        ["age", "sentence"],
        "sentence",
        "prediction",
        wer_metric,
        metric_mode="min",
        # feature_types={"age": "ordinal"},
        # feature_orders={"age": ["", "teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"]},
        min_support=10,
    )

    df["age"] = df["age"].astype("category")
    df["gender"] = df["gender"].astype("category")
    df["accent"] = df["accent"].astype("category")

    df = pd.concat((df, issue_df), axis=1)

    data_issue_severity = []
    data_issues = []
    for issue in issue_df["issue"].unique():
        if issue == -1:
            continue
        issue_rows = np.where(issue_df["issue"] == issue)[
            0
        ].tolist()  # Note: Has to be row index not pandas index!
        issue_metric = issue_df[issue_df["issue"] == issue].iloc[0]["issue_metric"]
        issue_explanation = (
            f"{issue_metric:.2f} -> "
            + issue_df[issue_df["issue"] == issue].iloc[0]["explanation"]
        )

        data_issue = DataIssue(
            severity="warning", description=issue_explanation, rows=issue_rows
        )
        data_issues.append(data_issue)
        data_issue_severity.append(issue_metric)

    data_issue_order = np.argsort(data_issue_severity)[::-1] # TODO Has to differ in max metric case

    spotlight.show(
        df,
        dtype={
            "speaker_embedding": Embedding,
            "text_embedding_ann": Embedding,
            "text_embedding_pred": Embedding,
        },
        issues=np.array(data_issues)[data_issue_severity].tolist(),
    )


if __name__ == "__main__":
    test_segment_guard()
