from os import rename
from typing import List
from pathlib import Path
import pandas as pd
import datasets
from datasets import Image, Audio, ClassLabel, Value, Sequence
import uuid
import puremagic


def _get_tutorial_imports():
    try:
        from bing_image_downloader import downloader
    except ImportError:
        raise RuntimeError(
            'Optional dependency bing-image-downloader required! (run pip install "bing-image-downloader")'
        )
    return downloader


def write_file(data: dict, suffix: str, data_dir: str):
    with open(f"{data_dir}/{uuid.uuid4().hex}{suffix}", "wb") as tmp:
        tmp.write(data["bytes"])
        return tmp.name


def convert_data(data: dict, data_dir: str):
    """
    Prefer raw data over path
    """
    if "bytes" in data and data["bytes"] is not None:
        if len(data["bytes"]) > 0:
            suffix = puremagic.from_string(data["bytes"])
            return write_file(data, suffix, data_dir)

    if "path" in data and data["path"] is not None:
        if data["path"] != "":
            suffix = puremagic.from_file(data["path"])
            new_path = f"{data['path']}{suffix}"

            # In case of missing file extension
            rename(data["path"], new_path)

            return new_path


# Tested with the following datasets:
# Image:
# "mnist"
# "ceyda/smithsonian_butterflies"
# "GabrielVidal/dead-by-daylight-perks"

# Audio:
# "437aewuh/dog-dataset"
# "Gae8J/modeling"
# "ccmusic-database/piano_sound_quality"

# Text:
# "xtreme", "XNLI"
# "indonlp/indonlu", "smsa"
# "tweet_eval", "emoji"


def from_huggingface(
    dataset_identifier: str,
    name=None,
    split=None,
    extract_dir="./sliceguard_tmp",
    force_redownload=False,
):
    # Simple utility method to support loading of huggingface datasets
    dataset = datasets.load_dataset(
        dataset_identifier,
        name,
        split,
        download_mode="force_redownload" if force_redownload else None,
    )

    overall_df = None

    # Create missing directories if non-existent
    Path(extract_dir).mkdir(parents=True, exist_ok=True)

    # Iterate splits in dataset.
    for split in dataset.keys():
        cur_split = dataset[split]

        split_df = dataset[split].to_pandas()
        split_df["split"] = split

        # Create a dataframe from each split.
        for fname, ftype in cur_split.features.items():
            if (
                not isinstance(ftype, Image)
                and not isinstance(ftype, Audio)
                and not isinstance(ftype, ClassLabel)
                and not isinstance(ftype, Value)
                and not isinstance(ftype, list)
                and not isinstance(ftype, Sequence)
            ):
                raise RuntimeError(
                    f"Found unsupported datatype {ftype}. Use custom load function."
                )

            if isinstance(ftype, list):
                split_df = split_df.drop(columns=fname)
                print(
                    f"Column {fname} with type {ftype} dropped. Lists are currently not supported."
                )

            # Run transformations for specific data types if needed.
            if isinstance(ftype, ClassLabel):
                class_label_lookup = {i: l for i, l in enumerate(ftype.names)}
                split_df[fname] = split_df[fname].map(lambda x: class_label_lookup[x])

            if isinstance(ftype, Image) or isinstance(ftype, Audio):
                if any(x is None for x in split_df[fname].values):
                    print("Column {fname} dropped due to None-type entries.")
                else:
                    split_df[fname] = split_df[fname].map(
                        lambda x: convert_data(x, extract_dir)
                    )

        if overall_df is None:
            overall_df = split_df
        else:
            overall_df = pd.concat((overall_df, split_df))

    return overall_df


def create_imagedataset_from_bing(
    queries: List[str],
    num_images: int = 10,
    dataset_folder: str = "./dataset",
    license: str = "All Creative Commons",
    adult_filter_off: bool = True,
    timeout: float = 1,
    test_split: float = None,
):
    """
    Download images from bing and create a folder structure that can be used with the ImageFolder dataset.
    :param queries: List of queries to download images for.
    :param num_images: Number of images to download per query.
    :param license: License to use for downloading images. Choose from: 'All', 'All Creative Commons',
        'Free to share and use', 'Free to share and use commercially',
        'Public Domain', 'Free to share and modify', 'Free to share and modify commercially'
    :param adult_filter_off: Whether to turn off adult filter.
    :param force_replace: Whether to force replace existing images.
    :param timeout: Timeout for downloading images.
    """
    downloader = _get_tutorial_imports()

    image_file_extensions = (".jpg", ".jpeg", ".png")

    if license is None:
        pass
    elif license == "All":
        pass
    elif license == "All Creative Commons":
        downloader.Bing.get_filter = lambda self, x: "filterui:licenseType-Any"

    elif license == "Public Domain":
        downloader.Bing.get_filter = lambda self, x: "filterui:license-L1"

    elif license == "Free to share and use":
        downloader.Bing.get_filter = (
            lambda self, x: "filterui:license-L2_L3_L4_L5_L6_L7"
        )
    elif license == "Free to share and use commercially":
        downloader.Bing.get_filter = lambda self, x: "filterui:license-L2_L3_L4"

    elif license == "Free to share and modify":
        downloader.Bing.get_filter = lambda self, x: "filterui:license-L2_L3_L5_L6"
    elif license == "Free to share and modify commercially":
        downloader.Bing.get_filter = lambda self, x: "filterui:license-L2_L3"
    else:
        raise ValueError(
            f"License {license} not supported. "
            "Choose from: 'All', 'All Creative Commons', 'Free to share and use', 'Free to share and use commercially',"
            "'Public Domain', 'Free to share and modify', 'Free to share and modify commercially'"
        )

    for query in queries:
        # check number of images in the folder and download if necessary
        downloaded_images_count = len(
            [
                p
                for p in (Path(dataset_folder) / query).glob("*.*")
                if p.suffix in image_file_extensions
            ]
        )
        if downloaded_images_count < num_images:
            downloader.download(
                query,
                limit=num_images - downloaded_images_count,
                output_dir=dataset_folder,
                adult_filter_off=adult_filter_off,
                force_replace=False,
                timeout=timeout,
            )
    df = pd.DataFrame()
    for i, query in enumerate(queries):
        image_paths = [
            str(p)
            for p in Path(f"{dataset_folder}/{query}/").glob("*.*")
            if p.suffix in image_file_extensions
        ]
        new_images = pd.DataFrame(
            {
                "image": image_paths[:num_images],
                "label_str": query,
                "label": i,
            }
        )
        df = pd.concat([df, new_images])

    df = df.reset_index(drop=True)

    if test_split is not None:
        df["split"] = "train"
        df.loc[df.sample(frac=test_split).index, "split"] = "val"

    return df
