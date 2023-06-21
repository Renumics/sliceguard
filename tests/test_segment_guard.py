import pandas as pd
import numpy as np
from jiwer import wer

from segment_guard import SegmentGuard


def wer_metric(y_true, y_pred):
    return np.mean([wer(s_y, s_pred) for s_y, s_pred in zip(y_true, y_pred)])

def test_segment_guard():
    df = pd.read_json("tests/predictions_embs.json")

    sg = SegmentGuard()
    sg.find_issues(df, ["sentence" ,"accent", "gender", "age"], "sentence", "pred", wer_metric, metric_mode="min")

    sg.report()

if __name__ == "__main__":
    test_segment_guard()