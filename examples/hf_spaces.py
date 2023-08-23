from pathlib import Path
import shutil
import json
import copy


def build_space(report_df, issues, output_dir, data_cols=[]):
    """
    A utility function to copy the data into a space compatible format.
    """
    report_df = report_df.copy()
    output_dir = Path(output_dir)

    for data_col in data_cols:
        data_path = output_dir / data_col
        if not data_path.is_dir():
            data_path.mkdir()
        new_paths = []
        for path in report_df[data_col]:
            path = Path(path)
            new_path = data_path / path.name
            shutil.copy(path, new_path)
            new_paths.append(str(new_path.relative_to(output_dir)))
        report_df[data_col] = new_paths

    df_path = output_dir / "df.json"
    report_df.to_json(df_path, orient="records")

    issues = copy.deepcopy(issues)
    for issue_index in range(len(issues)):
        issues[issue_index]["indices"] = issues[issue_index]["indices"].tolist()
        issues[issue_index]["rows"] = issues[issue_index]["rows"].tolist()

    issue_path = output_dir / "issues.json"
    with open(issue_path, "w") as f:
        json.dump(issues, f)
