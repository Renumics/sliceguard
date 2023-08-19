import pandas as pd
import datasets
from datasets import Image, ClassLabel


def from_huggingface(dataset_identifier: str):
    # Simple utility method to support loading of huggingface datasets
    # Currently only supports image data. Use custom load function if you need something else.
    dataset = datasets.load_dataset(dataset_identifier)
    overall_df = None
    for split in dataset.keys():
        cur_split = dataset[split]

        split_df = dataset[split].to_pandas()
        split_df["split"] = split

        for fname, ftype in cur_split.features.items():
            if not isinstance(ftype, Image) and not isinstance(ftype, ClassLabel):
                raise RuntimeError(
                    f"Found unsupported datatype {ftype}. Use custom load function."
                )
            if isinstance(ftype, ClassLabel):
                class_label_lookup = {i: l for i, l in enumerate(ftype.names)}
                split_df[fname] = split_df[fname].map(lambda x: class_label_lookup[x])

            if isinstance(ftype, Image):
                first_item = split_df[fname].iloc[0]
                if not "path" in first_item:
                    raise RuntimeError(
                        "Images are not extracted onto harddrive. Currently this is not supported."
                    )
                split_df[fname] = split_df[fname].map(lambda x: x["path"])

        if overall_df is None:
            overall_df = split_df
        else:
            overall_df = pd.concat((overall_df, split_df))

    return overall_df