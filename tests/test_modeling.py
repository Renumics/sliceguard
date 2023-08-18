from sklearn.metrics import accuracy_score
import datasets


from sliceguard import SliceGuard


def test_sliceguard_images():
    dataset = datasets.load_dataset('renumics/cifar100-enriched', split='all')
    df = dataset.to_pandas()

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df.sample(1000),
        ["image"],
        "fine_label",
        metric=accuracy_score,
        split_key="split",
        train_split="train",
        task='classification',
        min_drop=0.05,
        min_support=10
    )
    sg.report()

    # sg.report(spotlight_dtype={"image_path": Image})

if __name__ == "__main__":
    test_sliceguard_images()

