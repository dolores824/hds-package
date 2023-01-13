"""
This module contains a set of classes and functions 
for constructing classification models and making prediction.

Classes:
-SVMModel: 
Create a trained support vector machine model that can make prediction and give the prediction accuracy.
-RFModel:
Create a random forest model that can make prediction and give th prediction accuracy.

Functions
-cvd_risk_prediction
This function gives the information about whether the risk of developing cardiovascular disease for patients
with diabetes and the accuracy of the prediction.
-IgAN_risk_prediction
This function gives the information about whether the risk of developing Immunoglobulin A Nephropathy for
patients with diabetes and the accuracy of the prediction.
"""
from typing import List
from DiabetesComplecationPrediction.datasets import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df_cvd = diabetes_cvd_risk()
df_igan = diabetes_IgAN_risk()

class SVMModel():

    """
    Create a class of support vector machine models.

    The object in this class contains:
    -
    """

    def __init__(self, 
                 dataset, 
                 features: List[str], 
                 labels: str, 
                 input_data):
        """
        Create a support vector machine model.

        Args:
            dataset (DataFrame): the dataset used to train and test the model
            features (list): the name of the features used to train the model and predict the result
            labels (string): the name of the column which contains the labels used for model training
            input_data (list) : the information inputed by the patients
        """
        self.dataset = dataset
        self.features = features
        self.labels = labels
        self.X = self.dataset[self.features]
        self.y = self.dataset[self.labels]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 30)
        self.input_data = input_data

    def trained_model(self):
        """
        Generate and train a SVM model.

        Returns:
            trained SVM classification model
        """
        classifier = SVC()
        classifier.fit(self.X_train, self.y_train)
        return classifier

    def make_prediction(self, classifier):   
        """
        Use the trained model to predict the whether this person has risk of diabetes.

        Args:
            classifier: the trained model

        Returns:
            the prediction made by the classification model
        """   
        return classifier.predict(self.input_data)

    def model_accuracy(self):
        """
        Provide the accuracy of the classification model.

        Returns:
            the accuracy of the model without the percentage notation
        """
        accuracy = accuracy_score(self.y_test, self.trained_model().predict(self.X_test))
        return accuracy * 100

class RFModel():
    def __init__(self,
                 dataset,
                 features: List[str],
                 labels: str,
                 input_data):
        """
        Create a support vector machine model.

        Args:
            dataset (DataFrame): the dataset used to train and test the model
            features (list): the name of the features used to train the model and predict the result
            labels (string): the name of the column which contains the labels used for model training
            input_data (list) : the information inputed by the patients
        """
        self.dataset = dataset
        self.features = features
        self.labels = labels
        self.input_data = input_data
        self.X = self.dataset[self.features]
        self.y = self.dataset[self.labels]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 30)

    def trained_model(self):
        """
        Create and train a random forest model.

        Returns:
            trained RF classification model
        """
        classifier = RandomForestClassifier()
        classifier.fit(self.X_train, self.y_train)
        return classifier

    def make_prediction(self, classifier):
        """
        Use the trained model to predict the whether this person has risk of diabetes.

        Args:
            classifier: the trained model

        Returns:
            the prediction made by the classification model    
        """ 
        return classifier.predict(self.input_data)

    def model_accuracy(self):
        """
        Provide the accuracy of the classification model.

        Returns:
            the accuracy of the model without the percentage notation
        """
        accuracy = accuracy_score(self.y_test, self.trained_model().predict(self.X_test))
        return accuracy * 100
        

def cvd_risk_prediction(input, model_type = ['SVM', 'Random Forest']):
    """
    This function gives the prediction of whether the given information shows the patient is likely to 
    develop cardiovascular disease.

    Args:
        model_type (string): the choice of model for prediction
        input (dictionary): new information given by the patient
            key is the name of the features that the patient fills in
            value is the status of the specific feature
    Returns:
        a string that contains one sentence about the result for the prediction with the prediction accuracy
    """
    if model_type == 'SVM':
        model = SVMModel(df_cvd, input.keys(), 'Cardiovacular Risk', input.values())
    elif model_type == 'Random Forest':
        model = RFModel(df_cvd, input.keys(), 'Cardiovacular Risk', input.values())
    else: # Raise an error message to say please choose the model type within the given choice
        pass
    
    prediction = model.make_prediction(model.trained_model())
    if prediction == 1:
        print('It is likely to develop cardiovascular disease as a complication of diabetes.')
    else: 
        print('It is not likely to develop cadiovascular disease as a complication of diabetes.')

    print(f'The accuracy of this prediction is {model.model_accuracy()}%.')


def IgAN_risk_prediction(input, model_type = ['SVM', 'Random Forest']):
    """
    This function gives the prediction of whether the given information shows the patient is likely to develop
    IgAN as complication of diabetes.

    Args: 
        input (dictionary): new information given by the patient
            key is the name of the features that the patient fills in
            value is the corresponding status of that feature
        model_type (string): the choice of model for prediction
    Returns:
        a sentence that illustrate whether it is likely to have risk of getting IgAN according to the given
        information with the prediction accuracy
    """
    if model_type == 'SVM':
        model = SVMModel(df_igan, input.keys(), 'Risk of Nephropathy', input.values())
    elif model_type == 'Random Forest':
        model = RFModel(df_igan, input.keys(), 'Risk of Nephropathy', input.values())
    else:
        pass

    prediction = model.make_prediction(model.trained_model())
    if prediction == 1:
        print('The risk of developing Nephropathy as complication of diabetes is high.')
    else:
        print('The risk of developing Nephropathy as complication of diabetes is low.')

    print(f'The accuracy of this prediction is {model.model_accuracy()}%.')