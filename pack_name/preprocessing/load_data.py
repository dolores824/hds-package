import pandas as pd

def check_for_missing_values(self, entity_set, target_entity, column_name):
    """Checks if there is a missing value in the given column.
    Args:
        entity_set: fhir entityset.
        column_name: The column name to be checked for missing values.
        target_entity: The entity name which contains the column_name.
    Returns:
        False is the column_name does not contain a missing value.
    """
    if self.check_column_existence(entity_set, target_entity, column_name):

        nat = np.datetime64('NaT')
        missings = [
            nat,
            nan,
            'null',
            'nan',
            'NAN',
            'Nan',
            'NaN',
            'undefined',
            None,
            'unknown']
        contains_nan = False

        target_label_values = entity_set.__getitem__(target_entity).df[column_name]

        for missing_value in missings:
            if missing_value in list(target_label_values):
                contains_nan = True

        for missing_value in missings:
            for target_value in (target_label_values):
                if pd.isnull(target_value):
                    contains_nan = True

        return contains_nan
    else:
        return False

def load_csv(file_path):
    """
    Loads a csv file into a pandas dataframe. Checks for common null/missing values.
    Args:
        file_path (str): Full or relative path to file.
    Returns:
        (pandas.core.frame.DataFrame): The csv file in a dataframe
    """
    try:
        # Need to strip out whitespaces from the column names
        df = pd.read_csv(file_path, na_values=['None', 'null'])
        df = df.rename(columns=lambda x: x.strip())
        return df
    except FileNotFoundError:
        raise FileNotFoundError('No csv file was found at: {file_path}.\nPlease check your path and try again.')

