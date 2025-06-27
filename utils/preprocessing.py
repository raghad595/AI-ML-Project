import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
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
    
def scale_features(df, model="linear"):
    """
    Scale features in a DataFrame using specified method.
    Method determined based on outliers & Regression-Classification task.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    
    Returns:
    pd.DataFrame: DataFrame with features scaled.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_scaled = df.copy()
    # Check model type
    if model not in ["linear", "logistic"]:
        raise ValueError("Model type must be 'linear' or 'logistic'.")
    if model 
    has_outliers = False
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        col_std = df[col].std()
        # Avoid division by zero
        if col_std == 0:
            continue
        ratio = (col_max - col_min) / col_std
        if ratio > 10:
            has_outliers = True
            break
    # Choose scaling method based on outliers
    if has_outliers:
        scaler=StandardScaler()
    else:
        scaler=MinMaxScaler()
    # Scale only numeric columns
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled
    
def sample_data(df, features, target,n=5):
    """
    Sample a specified number of rows from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to sample from.
    n (int): Number of rows to sample.
    
    Returns:
    pd.DataFrame: Sampled DataFrame.
    """
    # Used for auto strategy selection in imbalanced datasets
    counts = Counter(target)
    majority_class = max(counts, key=counts.get)
    minority_class = min(counts, key=counts.get)
    imbalance_ratio = counts[majority_class] / counts[minority_class]
    # Balanced
    if imbalance_ratio < 1.5:
        pass
    #Oversampling
    elif imbalance_ratio < 3:
        over_sampler = RandomOverSampler(sampling_strategy="minority", random_state=42)
        x, y = over_sampler.fit_resample(features, target)
    #SMOTE
    elif imbalance_ratio < 5:
        smote = SMOTE(sampling_strategy="minority", random_state=42)
        x, y = smote.fit_resample(features, target)
    else:
        # Both oversampling and undersampling
        over_sampler = RandomOverSampler(sampling_strategy=0.4)
        x, y = over_sampler.fit_resample(features, target)
        under_sampler = RandomUnderSampler(sampling_strategy=0.5)
        x, y = under_sampler.fit_resample(x, y)
    return x, y

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