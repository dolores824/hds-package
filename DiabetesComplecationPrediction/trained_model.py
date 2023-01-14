"""
This module contains a set of classes and functions 
for constructing classification models and making prediction.

Classes:
-SVMModel: 
Create a trained support vector machine model that can make prediction and give the model evaluation.
-RFModel:
Create a random forest model that can make prediction and give the model evaluation.
-CatBoostModel:
Create a catboost model that can make prediction and give the model evaluation.

Functions
-cvd_risk_prediction
This function gives the information about whether the risk of developing cardiovascular disease for patients
with diabetes and the accuracy of the prediction.
-IgAN_risk_prediction
This function gives the information about whether the risk of developing Immunoglobulin A Nephropathy for
patients with diabetes and the accuracy of the prediction.
"""
from catboost import CatBoostClassifier
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DiabetesComplecationPrediction.datasets import *
from DiabetesComplecationPrediction.error import *
from DiabetesComplecationPrediction.preprocessing.missing import normal_impute, Del_Feature
from DiabetesComplecationPrediction.preprocessing.encode import encode_categ

del_feature = Del_Feature()

df_cvd = diabetes_cvd_risk()
df_igan = diabetes_IgAN_risk()

igan_missing = del_feature.missing_rate(df_igan)
df_CVD = normal_impute(df_cvd)
df_IgAN = normal_impute(del_feature.del_feature(df_igan, igan_missing))
df_IgAN = encode_categ(df_IgAN, 'Risk of Nephropathy')

class SVMModel():

    """
    Create a class of support vector machine models.
    """

    def __init__(self, 
                 dataset, 
                 features: List[str], 
                 labels: str):
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.30)

    def trained_model(self, kernel='linear', probability=True):
        """
        Generate and train a SVM model.

        Returns:
            trained SVM classification model
        """
        classifier = SVC(kernel=kernel, probability=probability)
        classifier.fit(self.X_train, self.y_train)
        return classifier

    def make_prediction(self, classifier, input_data):   
        """
        Use the trained model to predict the whether this person has risk of diabetes.

        Args:
            classifier: the trained model

        Returns:
            the prediction made by the classification model
        """   
        self.input_data = input_data
        return classifier.predict(self.input_data)

    def model_accuracy(self):
        """
        Provide the accuracy of the classification model.

        Returns:
            the accuracy of the model without the percentage notation
        """
        accuracy = accuracy_score(self.y_test, self.trained_model().predict(self.X_test))
        return accuracy * 100
    
    def roc(self):
        """
        Calculates the AUC score of the SVM model.

        Returns:
            float: the AUC score of the SVM model
        """
        clf = self.trained_model()
        y_scores = clf.predict_proba(self.X_test)[:,1]
        roc_score = roc_auc_score(self.y_test, y_scores)
        return roc_score

    def roc_plot(self):
        """
        Plots the ROC curve of the SVM model.
        The AUC score is included in the legend of the plot.
        """
        clf = self.trained_model()
        y_scores = clf.predict_proba(self.X_test)[:,1]
        roc_score = roc_auc_score(self.y_test, y_scores)
        fpr, tpr, _ = roc_curve(self.y_test, y_scores)

        # Plot ROC curve
        plt.plot(fpr, tpr, 
                color='darkorange',
                label='ROC curve (area = {:.3f})'.format(roc_score))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
        plt.show()


class RFModel():
    def __init__(self,
                 dataset,
                 features: List[str],
                 labels: str):
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.3)

    def trained_model(self):
        """
        Create and train a random forest model.

        Returns:
            trained RF classification model
        """
        classifier = RandomForestClassifier()
        classifier.fit(self.X_train, self.y_train)
        return classifier

    def make_prediction(self, classifier, input_data):
        """
        Use the trained model to predict the whether this person has risk of interests.

        Args:
            classifier: the trained model

        Returns:
            the prediction made by the classification model    
        """ 
        return classifier.predict(input_data)

    def model_accuracy(self):
        """
        Provide the accuracy of the classification model.

        Returns:
            the accuracy of the model without the percentage notation
        """
        accuracy = accuracy_score(self.y_test, self.trained_model().predict(self.X_test))
        return accuracy * 100

    def roc(self):
        """
        Calculates the AUC score of the random forest model.

        Returns:
            float: the AUC score of the model
        """
        clf = self.trained_model()
        y_scores = clf.predict_proba(self.X_test)[:,1]
        roc_score = roc_auc_score(self.y_test, y_scores)
        return roc_score

    def roc_plot(self):
        """
        Plots the ROC curve of the random forest model.
        The AUC score is included in the legend of the plot.
        """
        clf = self.trained_model()
        y_scores = clf.predict_proba(self.X_test)[:,1]
        roc_score = roc_auc_score(self.y_test, y_scores)
        fpr, tpr, _ = roc_curve(self.y_test, y_scores)

        # Plot ROC curve
        plt.plot(fpr, tpr, 
                color='darkorange',
                label='ROC curve (area = {:.3f})'.format(roc_score))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
        plt.show()

class CatBoostModel():
    def __init__(self, df, features, target):
        """
        Fit a CatBoost classification model.

        Args:
            df (pandas.DataFrame): the df used to train and test the model
            features (list): the name of the features used to train the model and predict the result
            target (string): the name of the column which contains the labels used for model training

        """
        self.df = df
        self.features = features
        self.target = target
        self.X = self.df[self.features]
        self.y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3)
        
    def trained_model(self):
        """
        Fit a CatBoost model

        Returns:
            trained CatBoost model
        """
        model = CatBoostClassifier()
        model.fit(self.X_train, self.y_train)
        return model

    def predict(self, model, input_data):
        """
        Use the catboost model to predict whether this person has risk of interests.

        Args:
            model: the catboost model

        Returns:
            the prediction made by the classification model  
        """
        return model.predict(input_data)

    def model_accuracy(self, model):
        """
        Provide the accuracy of catboost model.

        Args:
            model: the catboost model

        Returns:
            the accuracy of the model without the percentage notation
        """
        accuracy = accuracy_score(self.y_test, model.predict(self.X_test))
        return accuracy * 100
    
    def roc(self, model):
        """
        Compute the AUC score for Catboost model

        Args:
            model: the catboost model

        Returns:
            float: AUC score 
        """
        y_scores = model.predict_proba(self.X_test)[:, 1]
        roc_score = roc_auc_score(self.y_test, y_scores)
        return roc_score

    def roc_plot(self, model):
        """
        Plots the ROC curve of the SVM model.
        The AUC score is included in the legend of the plot.

        Args:
            model: the catboost model
        """
        y_scores = model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_scores)
        plt.plot(fpr, tpr, 
                color='darkorange',
                label='ROC curve (area = {:.3f})'.format(self.roc(model)))        
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
        plt.show()
        

def cvd_risk_prediction(input, model_type = ['SVM', 'Random Forest']):
    """
    This function gives the prediction of whether the given information shows the patient is likely to 
    develop cardiovascular disease.

    Args:
        model_type (string): the choice of model for prediction
                             options are 'SVM' and 'Random Forest'
        input (dictionary): new information given by the patient
            key is the name of the features that the patient fills in
            value is the status of the specific feature
    Returns:
        a string that contains one sentence about the result for the prediction with the prediction accuracy
    """
    features_list = list(input.keys())
    if model_type == 'SVM':
        model = SVMModel(dataset = df_CVD, features = features_list, labels = 'Cardiovascular Risk')
    elif model_type == 'Random Forest':
        model = RFModel(dataset = df_CVD, features = features_list, labels = 'Cardiovascular Risk')
    else: 
        raise DiaCcsPredError('Please choose a valid model type: "SVM" or "Random Forest".')
    
    input_list = [list(input.values())]
    prediction = model.make_prediction(model.trained_model(), input_list)
    if prediction == 1:
        print('It is likely to develop cardiovascular disease as a complication of diabetes.')
    else: 
        print('It is not likely to develop cadiovascular disease as a complication of diabetes.')

    print('The accuracy of this prediction is {:.2f}%.'.format(model.model_accuracy()))


def IgAN_risk_prediction(input, model_type = ['SVM', 'Random Forest']):
    """
    This function gives the prediction of whether the given information shows the patient is likely to develop
    IgAN as complication of diabetes.

    Args: 
        input (dictionary): new information given by the patient
            key is the name of the features that the patient fills in
            value is the corresponding status of that feature
        model_type (string): the choice of model for prediction
            options are 'SVM' and 'Random Forest'
    Returns:
        a sentence that illustrate whether it is likely to have risk of getting IgAN according to the given
        information with the prediction accuracy
    """
    feature_list = list(input.keys())
    if model_type == 'SVM':
        model = SVMModel(df_IgAN, feature_list, 'Risk of Nephropathy')
    elif model_type == 'Random Forest':
        model = RFModel(df_IgAN, feature_list, 'Risk of Nephropathy')
    else:
        raise DiaCcsPredError('Please choose a valid model type: "SVM" or "Random Forest".')

    input_list = [list(input.values())]
    prediction = model.make_prediction(model.trained_model(), input_list)
    if prediction == 1:
        print('The risk of developing Nephropathy as complication of diabetes is high.')
    else:
        print('The risk of developing Nephropathy as complication of diabetes is low.')

    print('The accuracy of this prediction is {:.2f}%.'.format(model.model_accuracy()))