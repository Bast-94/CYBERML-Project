import pandas as pd

def get_column_names(dataframe: pd.DataFrame) -> list[str]:
    """
    Get the name of columns in the dataframe

    Args:
        dataframe (pd.DataFrame): input dataframe
    Returns:
        list[str]: name of columns
    """
    return dataframe.columns.tolist()

def get_nb_of_dimensions(dataframe: pd.DataFrame) -> int:
    """
    Retrieves the number of dimensions of a pandas dataframe
 
    Args:
        dataframe (pd.DataFrame): input dataframe
    Returns:
        int: number of dimensions
    """
    if dataframe is None:
        return None
    
    return dataframe.columns.shape[0]

def get_nb_of_rows(dataframe: pd.DataFrame) -> int:
    """
    Get the number of rows
 
    Args:
        dataframe (pd.DataFrame): input dataframe
    Returns:
        int: number of rows
    """
    if dataframe is None:
        return None
    return dataframe.shape[0]

def get_number_column_names(dataframe: pd.DataFrame) -> list[str]:
    """
    Get the number of object columns
 
    Args:
        dataframe (pd.DataFrame): input dataframe
    Returns:
        list[str]: name of object columns
    """
    if dataframe is None:
        return None
    return dataframe.select_dtypes(include='number').columns.tolist()

def get_object_column_names(dataframe: pd.DataFrame) -> list[str]:
    """
    Get the name of object columns
 
    Args:
        dataframe (pd.DataFrame): input dataframe
    Returns:
        list[str]: name of object columns
    """
    if dataframe is None:
        return None
    return dataframe.select_dtypes(include='object').columns.tolist()

def get_unique_values(dataframe: pd.DataFrame, column_name: str) -> list[object]:
    """
    Get the unique values for a given column
 
    Args:
        dataframe (pd.DataFrame): input dataframe
        column_name (str): target column label
    Returns:
        list[object]: unique values for a given column
    """
    if dataframe is None:
        return None
    return dataframe[column_name].unique().tolist()