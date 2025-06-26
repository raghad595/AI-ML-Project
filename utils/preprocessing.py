import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Handling missing values in a DataFrame: Remove, Imputation, or Fill
def fill_missing_values(df, method='mean', value=None):
    """
    Fill missing values in a DataFrame using specified method or value.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    method (str): Method to fill missing values ('mean', 'median', 'mode', 'constant').
    value (any): Value to use if method is 'constant'.
    
    Returns:
    pd.DataFrame: DataFrame with missing values filled.
    """
    if method == 'mean':
        return df.fillna(df.mean())
    elif method == 'median':
        return df.fillna(df.median())
    elif method == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif method == 'ffill':
        return df.fillna(method='ffill')
    elif method == 'bfill':
        return df.fillna(method='bfill')
    elif method == 'constant':
        if value is not None:
            return df.fillna(value)
        else:
            raise ValueError("Value must be provided for constant filling.")
    else:
        raise ValueError("Invalid method specified.")
    
def remove_missing_values(df):
    """
    Remove rows with any missing values from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    
    Returns:
    pd.DataFrame: DataFrame with rows containing missing values removed.
    """
    return df.dropna()

def encode_categorical_features(df, method='onehot'):
    """
    Encode categorical features in a DataFrame using specified method.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    method (str): Encoding method ('onehot', 'label').
    
    Returns:
    pd.DataFrame: DataFrame with categorical features encoded.
    """
    if method == 'onehot':
        return pd.get_dummies(df, drop_first=True)
    elif method == 'label':
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col])
        return df
    else:
        raise ValueError("Invalid encoding method specified.")
    
def scale_features(df, method='minmax'):
    """
    Scale features in a DataFrame using specified method.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    method (str): Scaling method ('minmax', 'standard').
    
    Returns:
    pd.DataFrame: DataFrame with features scaled.
    """
    if method == 'minmax':
        return (df - df.min()) / (df.max() - df.min())
    elif method == 'standard':
        return (df - df.mean()) / df.std()
    else:
        raise ValueError("Invalid scaling method specified.")
    

def preprocess_data(df, fill_method='mean', encode_method='onehot', scale_method='minmax'):
    """
    Preprocess the DataFrame by filling missing values, encoding categorical features, and scaling features.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    fill_method (str): Method to fill missing values.
    encode_method (str): Method to encode categorical features.
    scale_method (str): Method to scale features.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    df = fill_missing_values(df, method=fill_method)
    df = encode_categorical_features(df, method=encode_method)
    df = scale_features(df, method=scale_method)
    return df