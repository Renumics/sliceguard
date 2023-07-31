import numpy as np
from sklearn.ensemble import IsolationForest


def fit_outlier_detection_model(encoded_data: np.array):
    clf = IsolationForest()
    clf.fit(encoded_data)
    outlier_scores = -clf.score_samples(encoded_data)
    return outlier_scores
