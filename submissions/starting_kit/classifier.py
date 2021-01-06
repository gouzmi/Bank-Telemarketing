from sklearn.base import BaseEstimator

class Classifier(BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        return None
    
    def predict_proba(self, X):
        return None
