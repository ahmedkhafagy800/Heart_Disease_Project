# ضع هذا في أعلى app.py قبل تحميل الموديل
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSlicer(BaseEstimator, TransformerMixin):
    def __init__(self, indices):
        self.indices = np.array(indices)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, self.indices]
