import numpy as np
import pandas as pd
import sys

from sklearn.base import BaseEstimator
from sklearn.impute import KNNImputer
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class FeatureExtractor(BaseEstimator):
    def __init__(self, imputer_neighbors: int = 5):
        self.imputer = KNNImputer(n_neighbors=imputer_neighbors)
        self.cat_cols = None
        self.num_cols = None

    def fit(self, X, y=None):
        # convert categorical columns to categorical type
        self.cat_cols = [
            column_name for column_name in X.columns
            if str(X[column_name].dtype) == 'object'
        ]
        self.num_cols = [
            column_name for column_name in X.columns
            if column_name not in self.cat_cols
        ]
        X[self.cat_cols] = X[self.cat_cols].astype('category')

        # one hot encode to be able to use KNNImputation
        X_dummy = X.copy()
        X_dummy = pd.get_dummies(X, dummy_na=True)
        for col in self.cat_cols:
            X_dummy.loc[X_dummy[col + "_nan"] == 1,
                        X_dummy.columns.str.startswith(col)] = np.nan
            del X_dummy[col + "_nan"]

        # fit imputer
        self.imputer.fit(X_dummy)

    def transform(self, X):
        # one hot encode to be able to use KNNImputation
        X_dummy = X.copy()
        X_dummy = pd.get_dummies(X, dummy_na=True)
        for col in self.cat_cols:
            X_dummy.loc[X_dummy[col + "_nan"] == 1,
                        X_dummy.columns.str.startswith(col)] = np.nan
            del X_dummy[col + "_nan"]

        X_dummy = pd.DataFrame(self.imputer.transform(X_dummy.values),
                               columns=X_dummy.columns)

        # revert dummification
        for col in self.cat_cols:
            X_dummy[col] = X_dummy.loc[:,
                                       X_dummy.columns.str.
                                       startswith(col)].idxmax(
                                           axis=1).str.replace(col + "_", '')
            X_dummy = X_dummy.loc[:,
                                  ~X_dummy.columns.str.startswith(col + "_")]

        # reset categorical column types
        X_dummy[self.cat_cols] = X_dummy[self.cat_cols].astype('category')

        # simplify pdays & previous
        X_dummy.pdays = np.where(X_dummy.pdays != 999., 1, 0)
        X_dummy.previous = np.where(X_dummy.previous >= 1., 1, 0)
        X_dummy.drop(columns=['previous','loan'], inplace=True)

        return X_dummy