"""
This module contains some functions to collect the input information and illustrate the results of prediction.

Functions:
- cvd_checker_csv
Take as input a DataFrame and a model choice, and returns a likelihood prediction of developing cardiovascular disease (CVD).

- IgAN_checker_csv
Take as input a DataFrame and a model choice, and returns a likelihood prediction of developing Immunoglobulin A Nephropathy (IgAN).
"""
from DiabetesComplecationPrediction.trained_model import cvd_risk_prediction, IgAN_risk_prediction
from DiabetesComplecationPrediction.error import *

def cvd_checker_csv(input_dataset, classification_method = ['SVM', 'Random Forest', 'CatBoost']):
    """
    This function takes a dataframe as input and returns the prediction using chosen classification method.

    Args:
        input_dataset (DataFrame): the information inputted by the patient who want to get a general prediction
        classification_method (string): the name of the classification method (SVM, random forest or catboost)

    Returns:
        A brief description of cvd risk and accuracy
    """
    info = input_dataset.to_dict('index')['Values']
    if classification_method == 'SVM':
        return cvd_risk_prediction(info, 'SVM')
    elif classification_method == 'Random Forest':
        return cvd_risk_prediction(info, 'Random Forest')
    elif classification_method == 'CatBoost':
        return cvd_risk_prediction(info, 'CatBoost')
    else:
        raise DiaCcsPredError('Please choose a valid model type: SVM, Random Forest or CatBoost.')
    
def IgAN_checker_csv(input_dataset, classification_method = ['SVM', 'Random Forest', 'CatBoost']):
    """
    This function takes a dataframe as input and returns the prediction using chosen classification method.

    Args:
        input_dataset (DataFrame): the information inputted by the patient who want to get a general prediction
        classification_method (string): the name of the classification method (SVM, random forest or catboost)

    Returns:
        A brief description of Igan risk and accuracy
    """
    info = input_dataset.to_dict('index')['Values']
    if classification_method == 'SVM':
        return IgAN_risk_prediction(info, 'SVM')
    elif classification_method == 'Random Forest':
        return IgAN_risk_prediction(info, 'Random Forest')
    elif classification_method == 'CatBoost':
        return IgAN_risk_prediction(info, 'CatBoost')
    else:
        raise DiaCcsPredError('Please choose a valid model type: SVM, Random Forest CatBoost.')

