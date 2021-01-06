import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorClassifier
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import KFold

problem_title = "Step Detection with Inertial Measurement Units"
_target_column_name = 'y'
_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

workflow = rw.workflows.FeatureExtractorClassifier()

score_types = [
    rw.score_types.F1Above(name='F-score (loan acceptance)', precision=4),
]
# class Bank(FeatureExtractorClassifier):
#     def __init__(self, workflow_element_names=[
#             'feature_extractor', 'classifier', 'train.csv']):
#         super(Bank, self).__init__(workflow_element_names[:2])
#         self.element_names = workflow_element_names

# define the score (specific score for the FAN problem)

# class FScoreBank(BaseScoreType):
#     is_lower_the_better = False
#     minimum = 0.0
#     maximum = 1.0

#     def __init__(self, name='bank error', precision=2):
#         self.name = name
#         self.precision = precision

#     def __call__(self, y_true, y_pred):
#         if isinstance(y_true, pd.Series):
#             y_true = y_true.values
#         scores = [y_true[i] == y_pred[i] for i in range(len(y_true))]
#         loss = scores.count(False)/len(y_true)*100

#         return loss


def get_cv(X, y):
    cv = KFold(n_splits=8, random_state=42)
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
