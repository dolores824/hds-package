"""
This module contains functions for encoding categorical variables in a pandas DataFrame.

Functions:

- encode_categ(df, col_name):
Encodes a categorical column in a DataFrame using a LabelEncoder from the scikit-learn library.
"""



import pandas as pd
from sklearn import preprocessing
from sklearn import utils

def encode_categ(df, col_name):
    """
    Encode a categorical column in a DataFrame using a LabelEncoder from the scikit-learn library.

    Args:
        df (pandas DataFrame): DataFrame containing the categorical column
        col_name (str): name of the categorical column to be encoded

    Returns:
        pandas DataFrame: input DataFrame with the encoded column

    Usage:
        df = encode_categ(df, col_name)
    """
    lab = preprocessing.LabelEncoder()
    df[col_name] = lab.fit_transform(df[col_name])
    return df