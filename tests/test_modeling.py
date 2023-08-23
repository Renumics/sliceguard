from sklearn.metrics import accuracy_score
import datasets

from sliceguard import SliceGuard

def test_sliceguard_images():
    dataset = datasets.load_dataset("renumics/cifar100-enriched", split="all")
    df = dataset.to_pandas()

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df.sample(10000),
        ["image"],
        "fine_label_str",
        min_drop=0.05,
        min_support=10,
        metric=accuracy_score,
        metric_mode="max",
        automl_split_key="split",
        automl_train_split="train",
        automl_task="classification",
        # automl_use_full_embeddings=True,
        automl_time_budget=40.0,
    )
    sg.report()

    # sg.report(spotlight_dtype={"image_path": Image})


if __name__ == "__main__":
    test_sliceguard_images()
