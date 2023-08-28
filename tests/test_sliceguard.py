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
from renumics.spotlight import Image, Audio

from sliceguard import SliceGuard


def wer_metric(y_true, y_pred):
    return np.mean([wer(s_y, s_pred) for s_y, s_pred in zip(y_true, y_pred)])


def test_sliceguard_text():
    df = pd.read_json("tests/predictions.json")

    # df = df.drop(
    #     columns=["up_votes", "down_votes", "locale", "segment", "variant", "audio"]
    # )
    # df.sample(500).to_json("example_data.json", orient="records")

    df = df[df["accent"] != ""]
    df = df[df["age"] != ""]
    df = df[df["gender"] != ""]

    df["age"] = pd.Categorical(
        df["age"].values,
        ordered=True,
        categories=[
            "teens",
            "twenties",
            "thirties",
            "fourties",
            "fifties",
            "sixties",
            "seventies",
            "eighties",
            "nineties",
        ],
    )

    # df["gender"] = df["gender"].astype("category")
    # df["accent"] = df["accent"].astype("category")

    df["audio"] = "tests/" + df["audio"]

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df.sample(200),
        ["accent", "age", "sentence"],
        "sentence",
        "prediction",
        wer_metric,
        metric_mode="min",
        # feature_types={"age": "ordinal"},
        # feature_orders={
        #     "age": [
        #         "teens",
        #         "twenties",
        #         "thirties",
        #         "fourties",
        #         "fifties",
        #         "sixties",
        #         "seventies",
        #         "eighties",
        #         "nineties",
        #     ]
        # },
        # min_support=30,
        # min_drop=0.04,
    )

    # sg.report(spotlight_dtype={"audio": Audio})

    # sg._plot_clustering_overview()


def test_sliceguard_images():
    dataset = datasets.load_dataset("olivierdehaene/xkcd", split="train")
    df = dataset.to_pandas()
    image_urls = df["image_url"]
    data_dir = Path("test_data/xkcd")
    # Download the image data
    if not data_dir.is_dir():
        data_dir.mkdir(parents=True)
        valid_indices = []
        target_paths = []
        for i, url in enumerate(image_urls):
            url_path = urlparse(url).path
            url_ext = os.path.splitext(url_path)[1]
            file_name = str(uuid.uuid4())
            target_path = data_dir / f"{file_name}{url_ext}"
            try:
                res = requests.get(url, stream=True)
                if res.status_code == 200:
                    with open(target_path, "wb") as f:
                        shutil.copyfileobj(res.raw, f)
                    valid_indices.append(i)
                    target_paths.append(str(target_path))
            except:
                pass
        df = df.iloc[valid_indices]
        df["image_path"] = target_paths
        df.to_json(data_dir / "metadata.json")
    df = pd.read_json(data_dir / "metadata.json")

    df = df[~pd.isnull(df["transcript"])]
    df = df[~pd.isnull(df["explanation"])]

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df,
        ["image_path"],
        "explanation",
        "transcript",
        wer_metric,
        metric_mode="min",
        # feature_types={"age": "ordinal"},
        # feature_orders={"age": ["", "teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"]},
        min_support=10,
        min_drop=0.08,
    )

    sg.report(spotlight_dtype={"image_path": Image})


def test_sliceguard_audio():
    dataset = datasets.load_dataset(
        "renumics/dcase23-task2-enriched", "dev", split="all", streaming=False
    )
    df = dataset.to_pandas().sample(200)
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

    # computed_embeddings = sg.embeddings

    # issue_df = sg.find_issues(
    #     df,
    #     ["path"],
    #     "label",
    #     "class",
    #     accuracy_score,
    #     metric_mode="max",
    #     # feature_types={"age": "ordinal"},
    #     # feature_orders={"age": ["", "teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"]},
    #     precomputed_embeddings=computed_embeddings,
    #     min_support=5,
    #     min_drop=0.1,
    # )

    sg.report(spotlight_dtype={"path": Audio})


if __name__ == "__main__":
    # test_sliceguard_text()
    # test_sliceguard_text()
    test_sliceguard_images()
    # test_sliceguard_audio()
