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
from sliceguard import data

from sliceguard import SliceGuard


def wer_metric(y_true, y_pred):
    return np.mean([wer(s_y, s_pred) for s_y, s_pred in zip(y_true, y_pred)])


def test_huggingface_mnist():
    df = data.from_huggingface("mnist")

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df.sample(100),
        ["image"],
        y="label",
        metric=accuracy_score,
        metric_mode="max",
        min_support=10,
        min_drop=0.08,
    )

    sg.report(spotlight_dtype={"image_path": Image})


def test_huggingface_butterflies():
    df = data.from_huggingface("ceyda/smithsonian_butterflies")

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df,
        ["image"],
        y="scientific_name",
        metric=accuracy_score,
        metric_mode="max",
        min_support=10,
        min_drop=0.08,
        automl_train_split="train",
        automl_task="classification",
        automl_time_budget=40.0,
    )

    sg.report(spotlight_dtype={"image_path": Image})


def test_huggingface_dead_by_daylight_perks():
    df = data.from_huggingface("GabrielVidal/dead-by-daylight-perks")

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df,
        ["image"],
        y="type",
        metric=accuracy_score,
        metric_mode="max",
        min_support=10,
        min_drop=0.08,
        automl_train_split="train",
        automl_task="classification",
        # automl_use_full_embeddings=True,
        automl_time_budget=40.0,
    )

    sg.report(spotlight_dtype={"image_path": Image})


def test_huggingface_dog_dataset():
    df = data.from_huggingface("437aewuh/dog-dataset")

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df.sample(200),
        ["audio"],
        "label",
        metric=accuracy_score,
        metric_mode="max",
        embedding_models={"path": "superb/wav2vec2-base-superb-sid"},
        min_support=5,
        min_drop=0.1,
    )
    sg.report(spotlight_dtype={"path": Audio})


def test_huggingface_modeling():
    df = data.from_huggingface("Gae8J/modeling")

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df.sample(200),
        ["audio"],
        "label",
        metric=accuracy_score,
        metric_mode="max",
        automl_train_split="train",
        automl_task="classification",
        automl_time_budget=40.0,
    )
    sg.report(spotlight_dtype={"path": Audio})


def test_huggingface_piano():
    df = data.from_huggingface("ccmusic-database/piano_sound_quality")

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df.sample(200),
        ["audio"],
        "label",
        metric=accuracy_score,
        metric_mode="max",
        automl_train_split="train",
        automl_task="classification",
        # automl_use_full_embeddings=True,
        automl_time_budget=40.0,
    )
    sg.report(spotlight_dtype={"path": Audio})


def test_huggingface_xtreme():
    df = data.from_huggingface("xtreme", "XNLI")
    sg = SliceGuard()
    issue_df = sg.find_issues(
            df.sample(1000),
            ['language'],
            "gold_label",
            metric=accuracy_score,
            min_drop=0.05,
            min_support=10,
            automl_task="classification",
            automl_time_budget=40.0,
        )
    sg.report()


def test_huggingface_indonlu():
    df = data.from_huggingface("indonlp/indonlu", "smsa")
    sg = SliceGuard()
    issue_df = sg.find_issues(
            df.sample(1000),
            ['text'],
            "label",
            metric=accuracy_score,
            min_drop=0.05,
            min_support=10,
            automl_train_split="train",
            automl_task="classification",
            automl_time_budget=40.0,
        )
    sg.report()


def test_huggingface_tweet_eval():
    df = data.from_huggingface("tweet_eval", "emoji")
    sg = SliceGuard()
    issue_df = sg.find_issues(
            df.sample(1000),
            ['text'],
            "label",
            metric=accuracy_score,
            # metric_mode="max",
            # wer_metric,
            # metric_mode="min",
            min_drop=0.05,
            min_support=10,
            # automl_split_key="",
            automl_train_split="train",
            automl_task="classification",
            # automl_use_full_embeddings=True,
            automl_time_budget=40.0,
        )
    sg.report()


# Image:
test_huggingface_mnist()
# test_huggingface_butterflies()
# test_huggingface_dead_by_daylight_perks()

# Audio:
# test_huggingface_dog_dataset()
# test_huggingface_modeling()
# test_huggingface_piano()

# Text:
# test_huggingface_xtreme()
# test_huggingface_indonlu()
# test_huggingface_tweet_eval()
