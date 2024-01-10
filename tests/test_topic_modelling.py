import pandas as pd
import pickle
from sliceguard import SliceGuard
from sklearn.metrics import f1_score

def convert_labels_to_numerical(df):
    df["numerical"] = [
        1 if label == "positive" else 0 if label == "negative" else -1
        for label in df["sentiment"]
    ]
    return df

# genug Daten für > 1 topics

df = pd.read_csv("test_data/IMDB Dataset.csv")[:50]
print(df.shape)
with open("test_data/doc_embeddings.pkl", "rb") as file:
    embeddings = pickle.load(file)
print(embeddings.shape)
embeddings = embeddings[:50]
print(embeddings.shape)

sg = SliceGuard()

df = convert_labels_to_numerical(df)
print(df.columns)
print(df.head())


issue_df = sg.find_issues(df, ["review"], y="numerical", metric=f1_score)
print(issue_df)
with open("test_data/test_issues_tm.pkl", "wb") as file:
    pickle.dump(issue_df, file)