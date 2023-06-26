import pandas as pd
import numpy as np
from jiwer import wer
from renumics import spotlight
from renumics.spotlight import Embedding

from segment_guard import SegmentGuard


def wer_metric(y_true, y_pred):
    return np.mean([wer(s_y, s_pred) for s_y, s_pred in zip(y_true, y_pred)])


def test_segment_guard():
    df = pd.read_json("tests/predictions_embs.json")

    print(df.columns)

    sg = SegmentGuard()
    issue_df = sg.find_issues(
        df,
        ["accent", "gender", "age", "up_votes"],
        "sentence",
        "prediction",
        wer_metric,
        metric_mode="min",
        feature_types={"age": "ordinal"},
        feature_orders={"age": ["", "teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"]},
        min_support = 50
    )

    df["age"] = df["age"].astype("category")
    df["gender"] = df["gender"].astype("category")
    df["accent"] = df["accent"].astype("category")

    df = pd.concat((df, issue_df), axis=1)
    # spotlight.show(df, dtype={"speaker_embedding": Embedding, "text_embedding_ann": Embedding, "text_embedding_pred": Embedding})

    


if __name__ == "__main__":
    test_segment_guard()
