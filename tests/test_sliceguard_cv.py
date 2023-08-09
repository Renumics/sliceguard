
from sklearn.metrics import accuracy_score
import pandas as pd
import datasets
from renumics.spotlight import Image

from sliceguard import SliceGuard

from cleanvision.imagelab import Imagelab


def cv_issues_cleanvision(df, image_name='image'):

    image_paths = df['image'].to_list()
    imagelab = Imagelab(filepaths=image_paths)
    imagelab.find_issues()

    df_cv=imagelab.issues.reset_index()

    return df_cv


def test_sliceguard_cv():
    
    dataset = datasets.load_dataset("renumics/cifar100-enriched", split="test")
    df = dataset.to_pandas()

    df_cv=cv_issues_cleanvision(df)
    #save df as parquet
    #df_cv.to_parquet('cifar100-enrichment-cv.parquet')
    #df_cv = pd.read_parquet('cifar100-enrichment-cv.parquet')

    df = pd.concat([df, df_cv], axis=1)

    features=['dark_score', 'low_information_score', 'light_score', 'blurry_score', 'fine_label']
    y_pred = 'fine_label_prediction'
    y = 'fine_label'
    error = 'fine_label_prediction_error'

    #input for sliceguard
    precomputed_embeddings = {'embedding': df['embedding'].to_numpy()}
    feature_types = {'dark_score': 'numeric', 'low_information_score': 'numeric', 
    'light_score': 'numeric', 'blurry_score': 'numeric', 'fine_label': 'nominal' }

    sg = SliceGuard()
    issue_df = sg.find_issues(
        df,
        features,
        y,
        y_pred,
        accuracy_score,
        precomputed_embeddings = precomputed_embeddings,
        metric_mode="max",
        feature_types={'fine_label': 'nominal'}
    )
    
    issue_df, issues = sg.report(spotlight_dtype={"image": Image})

    issue_df.to_parquet('cifar100-enrichment-sg.parquet')

    import pickle 

    baseline_accuracy = accuracy_score(df['fine_label'], df['fine_label_prediction'])

    for idx, issue in enumerate(issues):
        indices = issue.rows
        slice_accuracy = accuracy_score(df['fine_label'][indices], df['fine_label_prediction'][indices])
        
        if slice_accuracy < 0.75*baseline_accuracy:
            issues[idx].severity = "high"
       

    with open('sliceguard-issues.pkl', 'wb') as f:
        pickle.dump(issues, f)



if __name__ == "__main__":
    print("test_sliceguard_cv")
    test_sliceguard_cv()
