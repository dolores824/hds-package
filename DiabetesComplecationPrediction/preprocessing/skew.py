"""
This module contains functions dealing with skewed data. 

Functions:
- skewness(df):
It can be used to calculate the skewness of each feature in a pandas DataFrame.

- box_cox(df,skewness_size=0.75):
It can be used to fix skewed data.
"""

import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

def skewness(df):
    """Calculates the skewness of the numerical features in a pandas DataFrame.

    Args:
        df (pandas DataFrame): The DataFrame for which to calculate the skewness of the numerical features.

    Returns:
        pandas DataFrame: A DataFrame with the skewness of each numerical feature in the input DataFrame.
    """
    # Calculates the skewness of each numerical feature
    skewed_feats = df.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

    # Create a pandas DataFrame with the skewness values
    skewness = pd.DataFrame(skewed_feats)
    skewness.columns=['skewness']

    return skewness

def box_cox(df,skewness_size=0.75):
    """Transforms skewed numerical features of a pandas DataFrame using the Box-Cox method.

    Args:
        df (pandas DataFrame): The DataFrame with skewed numerical features.
        skewness_size (float, optional): Threshold for the skewness values above which a feature is considered skewed. . Defaults to 0.75.

    Returns:
        pandas DataFrame: The input DataFrame with the skewed numerical features transformed using the Box-Cox method.
    """
    numeric_feats = df.dtypes[df.dtypes != "object"].index

    skewed_feats = df[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > skewness_size]
    skewed_features = high_skew.index

    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))
    
    return df