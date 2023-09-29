import os
import uuid
import shutil
from pathlib import Path
from urllib.parse import urlparse

from sklearn.metrics import accuracy_score
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from jiwer import wer
import datasets
# from renumics.spotlight import Image, Audio
from sliceguard import data

from sliceguard import SliceGuard


def test_huggingface_audio():
    # df = pd.read_json("tests/predictions.json")
    # df = data.from_huggingface("ashraq/esc50")
    df = data.from_huggingface("renumics/dcase23-task2-enriched")
    # dataset = datasets.load_dataset("ashraq/esc50")

    # df = df.drop(
    #     columns=["up_votes", "down_votes", "locale", "segment", "variant", "audio"]
    # )
    # df.sample(500).to_json("example_data.json", orient="records")

    # df = df[df["accent"] != ""]
    # df = df[df["age"] != ""]
    # df = df[df["gender"] != ""]

    # df["audio"] = "tests/" + df["audio"]

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df,
        ["path"],
        "label",
        "class",
        accuracy_score,
        metric_mode="max",
        # feature_types={"age": "ordinal"},
        # feature_orders={"age": ["", "teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"]},
        embedding_models={"path": "superb/wav2vec2-base-superb-sid"},
        min_support=5,
        min_drop=0.1,
    )
    sg.report(spotlight_dtype={"path": Audio})

test_huggingface_audio()
