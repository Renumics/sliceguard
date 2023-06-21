import pandas as pd

from segment_guard import SegmentGuard


def test_segment_guard():
    df = pd.read_json("tests/predictions_embs.json")

    sg = SegmentGuard()
    sg.find_issues(df, ["accent", "gender", "age"], "wer", metric_mode="min")

    sg.report()

if __name__ == "__main__":
    test_segment_guard()