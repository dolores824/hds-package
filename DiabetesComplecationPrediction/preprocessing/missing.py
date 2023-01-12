"""
This module contains a set of classes and functions
that help preprocess and clean a dataset by dealing with missing values.

Functions:
- DataFrameDropNaN:
It can be used to remove rows that contain NaN or None value.

- Del_Feature:
It can be used to delete the feature whose missing rate is greater than the provided rate. 
It includes methods for computing the missing rate, visualizing it, and delete feature based on missing rate.

- normal_impute(df,col_name=None,imputation='mean'):
It fills in missing values with mean, median or mode. 

- interpolate_impute(df,col_name):
It fills missing value using interpolation. 

- knn_impute(df,k=5):
It fills missing value with nearest neighbours.

- rf_impute(df,na_col):
It fills the missing value with random forest regressor. 
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statistics

from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer

from DiabetesComplecationPrediction.error import DiaCcsPredError



class DropNaN(TransformerMixin):
    """Remove NaN values in a pandas DataFrame. Rows with NaN or None are removed.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Remove rows containing NaN or None value.
        """
        return X.dropna(axis=0, how='all', inplace=True)

    def check_na(self,X,y=None):
        """check if the DataFrame contains any NaN
        """
        print(X.isnull().sum())

class Del_Feature:

    def missing_rate(self,df):
        """Compute the missing rate of each feature in a pandas DataFrame.

        Args:
            df (pandas DataFrame): DataFrame containing the features to check.

        Returns:
            pandas DataFrame:  A DataFrame containing the missing rate of each feature.
        """
        df_na = (df.isnull().sum()/len(df))*100
        df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)        
        missing_df = pd.DataFrame(df_na)
        missing_df.columns=['missing_rate']
        return missing_df

    def missing_rate_hist(self,df):
        """Plot the histogram of missing rate for each feature in a pandas DataFrame.

        Args:
            df (pandas DataFrame): DataFrame containing the features to check.
        """
        df_na = (df.isnull().sum()/len(df))*100
        df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)
        plt.figure(figsize=(25,15))
        sns.barplot(x=df_na.index, y=df_na)
        plt.xlabel('Feature', fontsize=15)
        plt.ylabel('Missing rate', fontsize=15)

    def del_feature(self,df,missing_df,missing_rate=30):
        """Delete the feature whose missing rate is greater than the provided rate.

        Args:
            df (pandas DataFrame): The DataFrame from which features will be deleted.
            missing_df (pandas DataFrame): A DataFrame containing the missing rate of each feature.
            missing_rate (float, optional): The maximum allowed missing rate for a feature. 
                                            Defaults to 30.

        Returns:
            pandas DataFrame: The DataFrame with the features whose missing rate is greater than the provided rate removed.
        """
        missing_df=missing_df[missing_df['missing_rate']>missing_rate]
        df_drop=df.drop(list(missing_df.index),axis=1)
        return df_drop

def normal_impute(df,col_name=None,imputation='mean'):
    """Replaces missing values in a pandas DataFrame with the mean, median, or mode of the column.
    This function can be used to impute missing values of a specific column or the entire DataFrame.

    Args:
        df (pandas DataFrame): The DataFrame with missing values.
        col_name (str, optional): The name of the column with missing values. If not provided,
                                  missing values in the entire DataFrame will be imputed.
                                  Defaults to None.
        imputation (str, optional): Method of imputation. One of 'mean', 'median', 'mode'. 
                                    Defaults to 'mean'.

    Raises:
        DiaCcsPredError: An error message when the column name provided is invalid.

    Returns:
        pandas DataFrame: DataFrame with imputed values
    """
    # check if the column name is valid
    if col_name:
        if col_name in list(df.columns):
            if imputation=='mean':
                df[col_name] = df[col_name].replace(np.nan, df[col_name].mean())
            elif imputation=='median':
                df[col_name] = df[col_name].replace(np.NaN, df[col_name].median())
            elif imputation=='mode':
                df[col_name] = df[col_name].replace(np.NaN, statistics.mode(df[col_name]))
            return df
        else:
            raise DiaCcsPredError('Please check the column name.')
    else:
        # Imputing all the missing values in dataframe
        if imputation=='mean':
            df = df.fillna(df.mean())
        elif imputation=='median':
            df = df.fillna(df.median())
        elif imputation=='mode':
            df=df.fillna(df.mode().iloc[0])
        return df


def interpolate_impute(df,col_name):
    """Replaces missing values in a pandas DataFrame with interpolated values.

    Args:
        df (pandas dataframe): The DataFrame with missing values.
        col_name (str): The name of the column with missing values.

    Raises:
        DiaCcsPredError: An error message when the column name provided is invalid.

    Returns:
        dataframe: DataFrame with imputed values

    Usage:
        df = interpolate_impute(df, 'name')
    """
    # check if the column name is valid
    if col_name in list(df.columns):
        df[col_name]=df[col_name].interpolate()
        return df
    else:
        raise DiaCcsPredError('Please check the column name.')

# source
# https://zhuanlan.zhihu.com/p/268521157
def knn_impute(df,k=5):
    """Replaces missing values in a pandas DataFrame with values estimated by k-Nearest Neighbors (KNN) imputation.

    Args:
        df (pandas dataframe): The DataFrame with missing values. 
        n_neighbors (int, optional): The number of nearest rows to use for imputation (default: 5).

    Returns:
        dataframe: DataFrame with imputed values

    Usage:
        df = interpolate_impute(df, 'name')
    """
    knn_imt=KNNImputer(n_neighbors=k)
    imputation=knn_imt.fit_transform(df)
    return pd.DataFrame(imputation, columns=df.columns)

# source
# https://zhuanlan.zhihu.com/p/115103738
def rf_impute(df,na_col):
    """Replaces missing values in a pandas DataFrame with estimated by a random forest regressor.

    Args:
        df (pandas dataframe): The DataFrame with missing values.
        na_col (str): The name of the column with missing values.

    Raises:
        DiaCcsPredError: An error message when the column name provided is invalid.

    Returns:
        df (pandas dataframe): DataFrame with imputed values

    Usage:
        rf_impute(df,'name')
        df
    """
    # check if the column name is valid
    if na_col in list(df.columns):

        df_all_num=df.select_dtypes(include=[np.float64,np.int64])
        df_all_num=df.drop(labels=na_col,axis=1)  # all other numeric columns except target column
        df_nan=df.loc[:,na_col]  # the target column

        # split into train set and test set
        Ytrain = df_nan[df_nan.notnull()]
        Ytest = df_nan[df_nan.isnull()]
        Xtrain = df_all_num.iloc[Ytrain.index]
        Xtest = df_all_num.iloc[Ytest.index]

        # fit the random forest model
        rfc = RandomForestRegressor(n_estimators=100)
        rfc = rfc.fit(Xtrain, Ytrain)
        Ypredict = rfc.predict(Xtest)

        # impute predicted values into dataframe
        df_nan[df_nan.isnull()] = Ypredict

        return df
    else:
        raise DiaCcsPredError('Please check the column name.')