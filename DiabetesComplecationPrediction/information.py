"""
This module contains some functions to collect the input information and illustrate the results of prediction.
"""
from DiabetesComplecationPrediction.trained_model import cvd_risk_prediction, IgAN_risk_prediction
from DiabetesComplecationPrediction.error import *
import pandas as pd

def df_to_dict(df):
    """
    This function changes the format of patient's information from dataframe to dictionary.

    Args:
        df (DataFrame): information inputed by the patient

    Returns:
        A dictionary that has the information inputed by the patients:
            -keys of the dictionary are the features
            -values are the status of the corresponding features
    """
    dict = pd.Dataframe(df, 'index')
    del dict['Features']
    return dict

def cvd_checker_csv(input_dataset, classification_method = ['SVM', 'Random Forest']):
    info = input_dataset.to_dict('index')['Values']
    if classification_method == 'SVM':
        return cvd_risk_prediction(info, 'SVM')
    elif classification_method == 'Random Forest':
        return cvd_risk_prediction(info, 'Random Forest')
    else:
        raise DiaCcsPredError('Please choose a valid model type: SVM or Random Forest.')
    
def IgAN_checker_csv(input_dataset, classification_method = ['SVM', 'Random Forest']):
    info = input_dataset.to_dict()
    if classification_method == 'SVM':
        return IgAN_risk_prediction(info, 'SVM')
    elif classification_method == 'Random Forest':
        return IgAN_risk_prediction(info, 'Random Forest')
    else:
        raise DiaCcsPredError('Please choose a valid model type: SVM or Random Forest.')

