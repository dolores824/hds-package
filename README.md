# DiabetesComplecationPrediction
DiabetesComplecationPrediction is a package providing analysis framework for diabetes complecation predictions.
## Directory structure
📦DiabetesComplecationPrediction
 ├─ 📂datasets
 │ ├─ 📂data
 │ │ ├─ 📜hw_Cardiovascular.csv
 │ │ └─ 📜hw_Nephropathy.csv
 │ ├─ 📜base.py
 │ └─ 📜__init__.py
 ├─ 📂preprocessing
 │ ├─ 📜correlation.py
 │ ├─ 📜load.py
 │ ├─ 📜missing.py
 │ ├─ 📜skew.py
 │ └─ 📜__init__.py
 ├─ 📜error.py
 ├─ 📜requirements.txt
 ├─ 📜setup.py
 └─ 📜__init__.py

## Description and Features
- Data loading
- Data imputation
- Feature correlation and heatmap plotting
## Installation
### Using pip
`pip install https://github.com/dolores824/hds-package`
## Datasets
| Dataset             | Module                                                         |
| ------------------- | -------------------------------------------------------------- |
| Cardiovascular risk | DiabetesComplecationPrediction.datasets.data.hw_Cardiovascular |
| Nephropathy risk    | DiabetesComplecationPrediction.datasets.data.hw_Nephropathy    | 
## Functions
### Deletion methods
```python
from DiabetesComplecationPrediction.preprocessing.missing import DropNaN, Del_Feature
drop_na = DropNaN()
del_feature = Del_Feature()
```
| Name                    | Description                                                              |
| ----------------------- | ------------------------------------------------------------------------ |
| drop_na.transfrom       | Remove rows containing NaN or None value.                                |
| del_feature.del_feature | Delete the feature whose missing rate is greater than the provided rate. | 
### Imputation methods
```python
from DiabetesComplecationPrediction.preprocessing.missing import normalnormal_impute, interpolate_impute, knn_impute, rf_impute
```
| Name               | Description                                         |
| ------------------ | --------------------------------------------------- |
| normal_impute      | Simple imputer using mean, median and mode methods. |
| interpolate_impute | Fills missing value using interpolation.            |
| knn_impute         | Fills missing values with nearest neighbours.       |
| rf_impute          | Fills missing values with random forest regressor.  |
## References
Cardea/core.py at cdb79cb0bdf0332af1d8b28b6c074fbeb2aef9c1 · MLBazaar/Cardea (no date) GitHub. Available at: https://github.com/MLBazaar/Cardea (Accessed: 12 January 2023).
healthcareai-py/base.py at cb82b94990fb3046edccb3740ae5653adce70940 · HealthCatalyst/healthcareai-py (no date) GitHub. Available at: https://github.com/HealthCatalyst/healthcareai-py (Accessed: 12 January 2023).
PyHealth/usecase.rst at master · sunlabuiuc/PyHealth (no date) GitHub. Available at: https://github.com/sunlabuiuc/PyHealth (Accessed: 12 January 2023).
