import pandas as pd

from segment_guard import SegmentGuard


def test_segment_guard():
    df = pd.read_json("tests/predictions_embs.json")
    print(df.columns)

    sg = SegmentGuard()
    sg.find_issues(df)



if __name__ == "__main__":
    test_segment_guard()