from sklearn.base import BaseEstimator
import numpy as np

class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self
    
    def predict_proba(self, X):
        y_pred = np.zeros((len(X), 2))
        return y_pred
