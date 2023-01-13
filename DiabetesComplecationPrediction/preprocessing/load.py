"""
This submodule provides a function to load csv file into a pandas dataframe.

Functions:
- load_csv(file_path): 
Loads a csv file into a pandas dataframe and checks for common null/missing values.
"""
import pandas as pd

def load_csv(file_path):
    """
    Loads a csv file into a pandas dataframe. Checks for common null/missing values.
    Args:
        file_path (str): Full or relative path to file.
    Returns:
        (pandas DataFrame): The csv file in a dataframe
    """
    try:
        df = pd.read_csv(file_path, na_values=['None', 'null'])
        return df

    except:
        raise FileNotFoundError(
            f"No csv file was found at: {file_path}.\nPlease check your path and try again.")
