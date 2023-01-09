# todo complete module desc
"""
This module contains functions to deal with missing data:
- 
"""

import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestRegressor

SUPPORTED_IMPUTE_STRATEGY = ['MeanMode', 'RandomForest']

class DataFrameDropNaN(TransformerMixin):
    """Remove NaN values. Columns that are NaN or None are removed."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Uses pandas.DataFrame.dropna function where axis=1 is column action, and
        # how='all' requires all the values to be NaN or None to be removed.
        return X.dropna(axis=0, how='all', inplace=True)

class DataFrameImputer( TransformerMixin ):
    
    """
    Impute missing values in a dataframe.
    
    Parameters
    ----------
    impute : boolean, default=True
    	If True, imputation of missing value takes place.
    	If False, imputation of missing value doesn't happens.
    	
    verbose : boolean, default=True
    	Controls the verbosity.
    	If False : No text information will be shown about imputation of missing values
    	
    imputeStrategy : string, default='MeanMode'
    	It decides the technique to be used for imputation of missing values.
    	If imputeStrategy = 'MeanMode', Columns of dtype object or category 
        (assumed categorical) and imputed by the mode value of that column. 
        Columns of other types (assumed continuous) : by mean of column.
    	
        If imputeStrategy = 'RandomForest', Columns of dtype object or category 
         (assumed categorical) : imputed using RandomForestClassifier. 
         Columns of other types (assumed continuous) : imputed using RandomForestRegressor
    			
    
    tunedRandomForest : boolean, default=False
    	If set to True, RandomForestClassifier/RandomForestRegressor to be used for 
    	imputation of missing values are tuned using grid search and K-fold cross 
    	validation.
    	
    	Note:
    	If set to True, imputation process may take longer time depending upon size of 
    	dataframe and number of columns having missing values.
    	
    numeric_columns_as_categorical : List of type String, default=None
    	List of column names which are numeric(int/float) in dataframe, but by nature 
    	they are to be considered as categorical.
    	
    	For example:
    	There is a column JobCode( Levels : 1,2,3,4,5,6)
    	If there are missing values in JobCode column, panadas will by default convert 
    	this column into type float.
    	
    	
    	If numeric_columns_as_categorical=None
    	Missing values of this column will be imputed by Mean value of JobCode column.
    	type of 'JobCode' column will remain float. 
    	
        If numeric_columns_as_categorical=['JobCode']
        Missing values of this column will be imputed by mode value of JobCode column.
        Also final type of 'JobCode' column will be numpy.object 
						 			
    """
    def __init__(self, impute=True, verbose=True, imputeStrategy='MeanMode', tunedRandomForest=False, 
                 numeric_columns_as_categorical=None ):
        self.impute = impute
        self.object_columns = None
        self.fill = None
        self.verbose = verbose
        
        self.impute_Object = None
        self.imputeStrategy = imputeStrategy
        self.tunedRandomForest = tunedRandomForest
        self.numeric_columns_as_categorical = numeric_columns_as_categorical
        if self.numeric_columns_as_categorical is not None:
            if type(numeric_columns_as_categorical) is str:
                self.numeric_columns_as_categorical = [numeric_columns_as_categorical]
            elif type(numeric_columns_as_categorical) is list:
                self.numeric_columns_as_categorical = numeric_columns_as_categorical
            else:
                raise HealthcareAIError( "Please provide \'numeric_columns_as_categorical = {}\' parameter in string/list format (for single column) or in list format (for multiple columns)".format(numeric_columns_as_categorical) )
                 
        

    def fit(self, X, y=None):
        """
        Description:
        ------------
        
        If imputeStrategy is : 'MeanMode' / None
            Missing value to be imputed are calculated using Mean and Mode of corresponding columns.
            1. Columns specified in 'numeric_columns_as_categorical' are explicitly converted into dtype='object'
            2. Values to be imputed are calculated and stored in variable: self.fill 
            3. Later inside transform function, the same values will be filled in place of missing values.
        
        If imputeStrategy is : 'RandomForest'
            1. Class object of DataFrameImputerRandomForest is created
            2. fit function of DataFrameImputerRandomForest class is called.
        """
        
        if self.impute is False:
            return self
        
            
        if ( self.imputeStrategy=='MeanMode' or self.imputeStrategy==None ):
            
            if( self.numeric_columns_as_categorical is not None ):
                for col in self.numeric_columns_as_categorical:
                    if( col not in list(X.columns) ):
                        raise HealthcareAIError('Column = {} mentioned in numeric_columns_as_categorical is not present in dataframe'.format(col))
                    else:
                        X[col] = X[col].astype( dtype='object', copy=True, error='raise' )
    
            # Grab list of object column names before doing imputation
            self.object_columns = X.select_dtypes(include=['object']).columns.values
            
            num_nans = X.isnull().sum().sum()
            num_total = X.shape[0] * X.shape[1]
            percentage_imputed = num_nans / num_total * 100

            self.fill = pd.Series([X[c].value_counts().index[0]
                                   if X[c].dtype == np.dtype('O')
                                      or pd.api.types.is_categorical_dtype(X[c])
                                   else X[c].mean() for c in X], index=X.columns)

            if self.verbose:
                print("Percentage Imputed: %.2f%%" % percentage_imputed)
                print("Note: Impute will always happen on prediction dataframe, otherwise rows are dropped, and will lead "
                      "to missing predictions")

            # return self for scikit compatibility
            return self
        elif ( self.imputeStrategy=='RandomForest' ):
            self.impute_Object = DataFrameImputerRandomForest( tunedRandomForest=self.tunedRandomForest, 
                                                              numeric_columns_as_categorical=self.numeric_columns_as_categorical, 
                                                              impute=self.impute, verbose=self.verbose )
            self.impute_Object.fit(X)
            return self
        else:
            raise HealthcareAIError('A imputeStrategy must be one of these types: {}'.format(SUPPORTED_IMPUTE_STRATEGY))

            

    def transform(self, X, y=None):
        """
        Description:
        ------------
        
        If imputeStrategy is : 'MeanMode' / None
            Missing value to be imputed are calculated using Mean and Mode of corresponding columns.
            1. Missing values of dataframe are filled using self.fill variable(generated in fill() function )
            2. Columns specified in 'numeric_columns_as_categorical' are explicitly converted into dtype='object'
            3. Columns captured in 'self.object_columns' during fill() function are ensured to be of dtype='object'
            
        If imputeStrategy is : 'RandomForest'
            1. Already Class object of DataFrameImputerRandomForest is created during fill() function.
            2.. Now transform() function of DataFrameImputerRandomForest class is called.
        """
        
        # Return if not imputing
        if self.impute is False:
            return X
        
        if ( self.imputeStrategy=='MeanMode' or self.imputeStrategy==None ):
            result = X.fillna(self.fill)
            
            if( self.numeric_columns_as_categorical is not None ):
                for col in self.numeric_columns_as_categorical:
                    result[col] = result[col].astype( dtype='object', copy=True, error='raise' )

            for i in self.object_columns:
                if result[i].dtype not in ['object', 'category']:
                    result[i] = result[i].astype('object')

            return result
        elif ( self.imputeStrategy=='RandomForest' ):
            result = self.impute_Object.transform(X)
            return result
        else:
            raise HealthcareAIError('A imputeStrategy must be one of these types: {}'.format(SUPPORTED_IMPUTE_STRATEGY))