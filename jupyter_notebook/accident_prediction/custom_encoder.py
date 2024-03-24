# custom_transformers.py

from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.frequency_map = {}

    def fit(self, X, y=None):
        for column in X.columns:
            self.frequency_map[column] = X[column].value_counts(normalize=True)
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for column in X.columns:
            X_encoded[column] = X_encoded[column].map(self.frequency_map[column])
        return X_encoded
