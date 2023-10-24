import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def get_one_hot_encoded_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves the one hot encoded dataframe

    Args:
        dataframe (pd.DataFrame): input dataframe
    Returns:
        pd.DataFrame: the associated one hot encoded dataframe
    """
    if dataframe is None:
        return None

    return pd.get_dummies(dataframe, columns=dataframe.select_dtypes(include='object').columns.tolist())

def remove_nan_through_mean_imputation(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove NaN (Not a Number) entries through mean imputation

    Args:
        dataframe (pd.DataFrame): input dataframe
    Returns:
        pd.DataFrame: the dataframe with  NaN (Not a Number) entries replaced using mean imputation
    """
    if dataframe is None:
        return None

    # Replace NaN (Not a Number) entries through mean imputation
    return dataframe.fillna(dataframe.mean())