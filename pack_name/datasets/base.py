from ..preprocessing.load_data import load_csv

def diabetes_cvd_risk():
    """
    Loads the sample dataset to investigate the cardiovascular disease risk for diabetes patients.

    Note: The dataset contains the following columns: # todo

    Usage:
        diabetes=pack_name.load_diabetes()
    """
    return load_csv('data/hw_Cardiovascular.csv')


def diabetes_IgAN_risk():
    """
    Loads the sample dataset to investigate the Immunoglobulin A Nephropathy risk for diabetes patients.

    Note: The dataset contains the following columns: # todo

    Usage:
        diabetes=pack_name.load_diabetes()
    """
    return load_csv('data/hw_Nephropathy.csv')

diabetes_cvd_risk()
