from sklearn.base import BaseEstimator
import lightgbm as lgb
import numpy as np


class Classifier(BaseEstimator):
    def __init__(self):
        self.model = lgb.LGBMClassifier(is_unbalance=True, n_jobs=-1)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        y_pred = self.model.predict(X)
        return np.array([1 - y_pred, y_pred]).T