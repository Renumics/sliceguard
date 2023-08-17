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
import pickle

from sliceguard import SliceGuard


def test_sliceguard_images():
    dataset = datasets.load_dataset('renumics/cifar100-enriched', split='all')
    df = dataset.to_pandas()

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df.sample(100),
        ["image"],
        "fine_label",
        metric=accuracy_score,
        split_key='split',
        task='regression'
    )

    # sg.report(spotlight_dtype={"image_path": Image})

if __name__ == "__main__":
    test_sliceguard_images()

