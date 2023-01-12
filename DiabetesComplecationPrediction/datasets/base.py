"""
This module is used to load the sample datasets. 

Functions:
- load_csv():
A helper function to load the data from a csv file.

- diabetes_cvd_risk():
To load the sample dataset to investigate the cardiovascular disease risk for diabetes patients.

- diabetes_IgAN_risk():
To load the sample dataset to investigate the Immunoglobulin A Nephropathy risk for diabetes patients.
"""

from os.path import dirname
from os.path import join
import pandas as pd

def load_csv(data_file_name):
    """Loads data from module_path/datasets/data/data_file_name
    Args:
        data_file_name (str) : Name of csv file to be loaded from
                               module_path/datasets/data/data_file_name
    Returns:
        pandas DataFrame: A pandas dataframe.
    """
    file_path = join(dirname(__file__), 'data', data_file_name)
    return pd.read_csv(file_path, na_values=['None'])

def diabetes_cvd_risk():
    """Loads the sample dataset to investigate the cardiovascular disease risk for diabetes patients.

    Usage:
        df_cvd = diabetes_cvd_risk()
    """
    return load_csv('hw_Cardiovascular.csv')


def diabetes_IgAN_risk():
    """Loads the sample dataset to investigate the Immunoglobulin A Nephropathy risk for diabetes patients.

    Usage:
        df_igan = diabetes_IgAN_risk()
    """
    return load_csv('hw_Nephropathy.csv')
