import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorClassifier
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score

problem_title = "Step Detection with Inertial Measurement Units"
_target_column_name = 'y'
_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

workflow = rw.workflows.FeatureExtractorClassifier()

score_types = [
    rw.score_types.MacroAveragedRecall(name='Balanced Accuracy', precision=4),
]


class FScoreBank(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='balanced_accuracy', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        score = balanced_accuracy_score(y_true, y_pred)
        return score


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits = 8, test_size = 0.2)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), delimiter=';')
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
