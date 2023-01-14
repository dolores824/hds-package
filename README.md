# DiabetesComplecationPrediction
[Github repo](https://github.com/dolores824/hds-package)   
DiabetesComplecationPrediction is designed to predict the risk of diabetes complications, specifically cardiovascular disease and Nephropathy. It utilizes machine learning models, such as support vector machine, random forest and catboost, to analyze patient data and make predictions.    
This package is essential for medical practitioners as it allows them to identify high-risk patients and take preventative measures to reduce the likelihood of complications. Additionally, it helps medical facilities to make data-driven decisions, which can lead to better outcomes for patients and more efficient use of resources.
## Directory structure
```
📦DiabetesComplecationPrediction
 ┣━ 📂datasets
 ┃ ┣━ 📂data
 ┃ ┃ ┣━ 📜cvd_exm.csv
 ┃ ┃ ┣━ 📜hw_Cardiovascular.csv
 ┃ ┃ ┣━ 📜hw_Nephropathy.csv
 ┃ ┃ ┗━ 📜igan_exm.csv
 ┃ ┣━ 📜base.py
 ┃ ┗━ 📜__init__.py
 ┣━ 📂preprocessing
 ┃ ┣━ 📜correlation.py
 ┃ ┣━ 📜encode.py
 ┃ ┣━ 📜load.py
 ┃ ┣━ 📜missing.py
 ┃ ┣━ 📜skew.py
 ┃ ┗━ 📜__init__.py
 ┣━ 📜error.py
 ┣━ 📜information.py
 ┣━ 📜setup.py
 ┣━ 📜trained_model.py
 ┗━ 📜__init__.py
```
## Features
- Data loading
- Data imputation
- Feature selection and heatmap plotting
- Model generation and evaluation
- Model comparison
- Disease risk prediction
## Installation
### Using pip
`pip install https://github.com/dolores824/hds-package`
### Intall dependencies
#### Using pip
`pip install -r requirements.txt`
#### Using conda
`conda install -r requirements.txt`
## Example usage showcase
[Predict the cardiovascular disease risk for diabetes patients](https://github.com/dolores824/hds-package/blob/master/example_usage_cvd.ipynb)   
[Predict the Immunoglobulin A Nephropathy risk for diabetes patients](https://github.com/dolores824/hds-package/blob/master/example_usage_igan.ipynb)
## Functions
### Check missing value method
```python
from DiabetesComplecationPrediction.preprocessing.missing import DropNaN, Del_Feature
drop_na = DropNaN()
del_feature = Del_Feature()
```
| Name                          | Description                                                                |
| ----------------------------- | -------------------------------------------------------------------------- |
| drop_na.check_na              | Check if the DataFrame contains any NaN.                                   |
| del_feature.missing_rate      | Compute the missing rate of each feature in a pandas DataFrame.            |
| del_feature.missing_rate_hist | Plot the histogram of missing rate for each feature in a pandas DataFrame. | 
### Deletion method
```python
from DiabetesComplecationPrediction.preprocessing.missing import DropNaN, Del_Feature
drop_na = DropNaN()
del_feature = Del_Feature()
```
| Name                    | Description                                                              |
| ----------------------- | ------------------------------------------------------------------------ |
| drop_na.transfrom       | Remove rows containing NaN or None value.                                |
| del_feature.del_feature | Delete the feature whose missing rate is greater than the provided rate. | 
### Imputation method
```python
from DiabetesComplecationPrediction.preprocessing.missing import normalnormal_impute, interpolate_impute, knn_impute, rf_impute
```
| Name               | Description                                         |
| ------------------ | --------------------------------------------------- |
| normal_impute      | Simple imputer using mean, median and mode methods. |
| interpolate_impute | Fill missing value using interpolation.            |
| knn_impute         | Fill missing values with nearest neighbours.       |
| rf_impute          | Fill missing values with random forest regressor.  |
### Encode method
```python
from Diabetes.preprocessing.encode import encode_categ
```
| Name         | Description                                  |
| ------------ | -------------------------------------------- |
| encode_categ | Encodes a categorical column in a DataFrame. | 
### Correlation method
```python
from DiabetesComplecationPrediction.preprocessing.correlation import cor_heatmap, Feature
feature = Feature()
```
| Name                         | Description                                                                         |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| cor_heatmap                  | Plot a heatmap of correlation matrix of a DataFrame.                                |
| feature.most_correlated      | Return the most correlated k features.                                              |
| feature.feature_list         | Return either the most correlated features or all features to be used in the model. | 
| feature.view_correlations    | Return correlation matrix for selected columns in a DataFrame.                      |
| feature.most_related_heatmap | Plot heatmap of most related columns based on correlation matrix.                   |
### Model method
```python
from DiabetesComplecationPrediction.trained_model import SVMModel, RFModel, CatBoostModel
svm=SVMModel()
rf=RFModel()
cb=CatBoostModel()
```
#### SVM model
| Name                | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| svm.trained_model   | Generate and trains a SVM model.                                        |
| svm.make_prediction | Use SVM model to predict the whether this person has risk of interests. |
| svm.model_accuracy  | Provide the accuracy of SVM model.                                      |
| svm.roc             | Calculate the AUC score of the SVM model.                              |
| svm.roc_plot        | Plots the ROC curve of the SVM model.                                   | 
#### Random forest model
| Name                | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| rf.trained_model   | Generate and trains a random forest model.                                        |
| rf.make_prediction | Use random forest model to predict the whether this person has risk of interests. |
| rf.model_accuracy  | Provide the accuracy of random forest model.                                      |
| rf.roc             | Calculate the AUC score of the random forest model.                              |
| rf.roc_plot        | Plot the ROC curve of the random forest model.                                   | 
#### Catboost model
| Name                | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| cb.trained_model   | Generate and trains a catboost model.                                        |
| cb.make_prediction | Use catboost model to predict the whether this person has risk of interests. |
| cb.model_accuracy  | Provide the accuracy of catboost model.                                      |
| cb.roc             | Calculates the AUC score of the catboost model.                              |
| cb.roc_plot        | Plot the ROC curve of the catboost model.                                   | 
## Example datasets
| Dataset             | Module                                                         |
| ------------------- | -------------------------------------------------------------- |
| Cardiovascular risk | DiabetesComplecationPrediction.datasets.data.hw_Cardiovascular |
| Nephropathy risk    | DiabetesComplecationPrediction.datasets.data.hw_Nephropathy    | 
### Example dataset prediction
```python
from DiabetesComplecationPrediction.trained_model import cvd_risk_prediction, IgAN_risk_prediction
```
| Name                 | Description                                                                       |
| -------------------- | --------------------------------------------------------------------------------- |
| cvd_risk_prediction  | Predict the likelihood of cardiovascular disease from given information.          |
| IgAN_risk_prediction | Predict the likelihood of Immunoglobulin A Nephropathy from given information.    |
## References
Cardea/core.py at cdb79cb0bdf0332af1d8b28b6c074fbeb2aef9c1 · MLBazaar/Cardea (no date) GitHub. Available at: https://github.com/MLBazaar/Cardea (Accessed: 12 January 2023).   
healthcareai-py/base.py at cb82b94990fb3046edccb3740ae5653adce70940 · HealthCatalyst/healthcareai-py (no date) GitHub. Available at: https://github.com/HealthCatalyst/healthcareai-py (Accessed: 12 January 2023).   
PyHealth/usecase.rst at master · sunlabuiuc/PyHealth (no date) GitHub. Available at: https://github.com/sunlabuiuc/PyHealth (Accessed: 12 January 2023).   
