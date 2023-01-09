# todo complete module desc
"""
This module contains functions to deal with missing data:
- 
"""

import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestRegressor

class DataFrameDropNaN(TransformerMixin):
    """Remove NaN values. Columns that are NaN or None are removed."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Uses pandas.DataFrame.dropna function where axis=1 is column action, and
        # how='all' requires all the values to be NaN or None to be removed.
        return X.dropna(axis=1, how='all')