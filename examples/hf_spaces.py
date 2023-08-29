from pathlib import Path
import shutil
import json
import copy
import pandas as pd
import numpy as np


def build_space(report_df, issues, output_dir, data_cols=[]):
    """
    A utility function to copy the data into a space compatible format.
    """
    report_df = report_df.copy()
    output_dir = Path(output_dir)

    # Copy data and adjust paths
    for data_col in data_cols:
        data_path = output_dir / data_col
        if not data_path.is_dir():
            data_path.mkdir()
        new_paths = []
        for path in report_df[data_col]:
            if pd.notnull(path):
                path = Path(path)
                new_path = data_path / path.name
                shutil.copy(path, new_path)
                new_path = str(new_path.relative_to(output_dir))
            else:
                new_path = path
            new_paths.append(new_path)
        report_df[data_col] = new_paths

    # Transform numpy types to python types
    for col in report_df.columns:
        if pd.api.types.is_float_dtype(report_df[col].dtype):
            new_data = []
            for item in report_df[col]:
                if isinstance(item, np.generic):
                    new_item = item.item()
                else:
                    new_item = item
                new_data.append(new_item)
            report_df[col] = report_df[col].astype("float")
            report_df[col] = new_data

        if pd.api.types.is_object_dtype(report_df[col].dtype):
            new_data = []
            for item in report_df[col]:
                if isinstance(item, np.ndarray):
                    new_item = item.tolist()
                else:
                    new_item = item
                new_data.append(new_item)
            report_df[col] = report_df[col].astype("object")
            report_df[col] = new_data

    df_path = output_dir / "df.json"
    report_df.to_json(df_path, orient="records")

    issues = copy.deepcopy(issues)

    for issue_index in range(len(issues)):
        issues[issue_index]["indices"] = issues[issue_index]["indices"].tolist()
        issues[issue_index]["rows"] = issues[issue_index]["rows"].tolist()
        if isinstance(issues[issue_index]["metric"], np.generic):
            issues[issue_index]["metric"] = issues[issue_index]["metric"].item()
        for col_info_idx in range(len(issues[issue_index]["explanation"])):
            col_info = issues[issue_index]["explanation"][col_info_idx]
            new_col_info = {}
            for k, v in col_info.items():
                if isinstance(v, np.generic):
                    new_v = v.item()
                else:
                    new_v = v
                new_col_info[k] = new_v
            issues[issue_index]["explanation"][col_info_idx] = new_col_info

    issue_path = output_dir / "issues.json"
    with open(issue_path, "w") as f:
        json.dump(issues, f)
