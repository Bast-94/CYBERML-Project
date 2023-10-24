import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from mlsecu.data_exploration_utils import *
from mlsecu.data_preparation_utils import *

def get_list_of_attack_types(dataframe: pd.DataFrame) -> list[str]:
    """
    Retrieves the name of attack types of a pandas dataframe
 
    Args:
        dataframe (pd.DataFrame): input dataframe
    Returns:
        list[str]: the name of distinct attack types
    """
    if dataframe is None:
        return None

    return dataframe['attack_type'].unique().tolist()

def get_nb_of_attack_types(dataframe: pd.DataFrame) -> int:
    """
    Retrieves the number of distinct attack types of a pandas dataframe
 
    Args:
        dataframe (pd.DataFrame): input dataframe
    Returns:
        int: the number of distinct attack types
    """
    if dataframe is None:
        return None

    return len(dataframe['attack_type'].unique())

def get_list_of_if_outliers(dataframe: pd.DataFrame, outlier_fraction: float) -> list[int]:
    """
    Extract the list of outliers according to Isolation Forest algorithm
 
    Args:
        dataframe (pd.DataFrame): input dataframe
        outlier_fraction (float): rate of outliers to be extracted
    Returns:
        list[int]: list of outliers according to Isolation Forest algorithm
    """
    if dataframe is None:
        return None
    
    one_hot_encoded = get_one_hot_encoded_dataframe(dataframe)
    no_nan_df = remove_nan_through_mean_imputation(one_hot_encoded)
    
    iforest = IsolationForest(contamination=outlier_fraction, random_state=42)
    predictions = iforest.fit_predict(no_nan_df)
    indices = [i for i, x in enumerate(predictions) if x == -1]

    return indices

def get_list_of_lof_outliers(dataframe: pd.DataFrame, outlier_fraction: float) -> list[int]:
    """
    Extract the list of outliers according to Local Outlier Factor algorithm
 
    Args:
        dataframe (pd.DataFrame): input dataframe
        outlier_fraction (float): rate of outliers to be extracted
    Returns:
        list[int]: list of outliers according to Local Outlier Factor algorithm
    """
    if dataframe is None:
        return None
    
    one_hot_encoded = get_one_hot_encoded_dataframe(dataframe)
    no_nan_df = remove_nan_through_mean_imputation(one_hot_encoded)
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)
    predictions = lof.fit_predict(no_nan_df)
    indices = [i for i, x in enumerate(predictions) if x == -1]

    return indices

def get_list_of_parameters(dataframe: pd.DataFrame) -> list[str]:
    """
    Retrieves the list of parameters of a pandas dataframe
 
    Args:
        dataframe (pd.DataFrame): input dataframe
    Returns:
        list[str]: list of parameters
    """
    if dataframe is None:
        return None
    
    return dataframe.columns.tolist()

def get_nb_of_if_outliers(dataframe: pd.DataFrame, outlier_fraction: float) -> int:
    """
    Extract the number of outliers according to Isolation Forest algorithm
    
    Args:
        dataframe (pd.DataFrame): input dataframe
        outlier_fraction (float): rate of outliers to be extracted
    Returns:
        int: number of outliers according to Isolation Forest algorithm
    """
    if dataframe is None:
        return None
    
    return len(get_list_of_if_outliers(dataframe, outlier_fraction))

def get_nb_of_lof_outliers(dataframe: pd.DataFrame, outlier_fraction: float) -> int:
    """
    Extract the number of outliers according to Local Outlier Factor algorithm
 
    Args:
        dataframe (pd.DataFrame): input dataframe
        outlier_fraction (float): rate of outliers to be extracted
    Returns:
        int: number of outliers according to Local Outlier Factor algorithm
    """
    if dataframe is None:
        return None
    
    return len(get_list_of_lof_outliers(dataframe, outlier_fraction))

def get_nb_of_occurrences(dataframe: pd.DataFrame) -> int:
    """
    Retrieves the number of occurrences of a pandas dataframe
 
    Args:
        dataframe (pd.DataFrame): input dataframe
    Returns:
        int: number of occurrences
    """
    if dataframe is None:
        return None
    
    return dataframe.shape[0]

def get_nb_of_parameters(dataframe: pd.DataFrame) -> int:
    """
    Retrieves the number of parameters of a pandas dataframe
 
    Args:
        dataframe (pd.DataFrame): input dataframe
    Returns:
        int: number of parameters
    """
    if dataframe is None:
        return None
    
    return dataframe.shape[1]