import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
# Handling missing values in a DataFrame: Remove, Imputation, or Fill
def fill_missing_values(df, value=None, method="auto"):
    """
    Fill missing values in a DataFrame using specified method or value.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    method (str): Method to fill missing values ('mean', 'median', 'mode', 'constant', etc.).
    value (any): Value to use if method is 'constant'.
    
    Returns:
    pd.DataFrame: DataFrame with missing values filled.
    """
    df_cleaned = df.copy()
    # Check the method
    if method == 'auto':
        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            time_series = True
        else:
            time_series = False
        missing_ratio = df_cleaned.isnull().mean().max()
        # If more than 50% of the data is missing, drop the column
        if missing_ratio < 0.5:
            df_cleaned = df_cleaned.dropna(axis=1, thresh=int(0.5 * len(df_cleaned)))
        elif time_series:
            # If time series data, fill with forward fill
            return df_cleaned.fillna(method='ffill')
        else:
            # Fill numeric with median, categorical with mode
            for col in df_cleaned.columns:
                if df_cleaned[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode().iloc[0])
        return df_cleaned
    elif method == 'mean':
        return df_cleaned.fillna(df_cleaned.mean())
    elif method == 'median':
        return df_cleaned.fillna(df_cleaned.median())
    elif method == 'mode':
        return df_cleaned.fillna(df_cleaned.mode().iloc[0])
    elif method == 'ffill':
        return df_cleaned.fillna(method='ffill')
    elif method == 'bfill':
        return df_cleaned.fillna(method='bfill')
    elif method == 'constant':
        return df_cleaned.fillna(value)
    elif method == 'drop':
        # Drop rows with any missing values
        return df_cleaned.dropna()
    
def encode_categorical_features(df, method="auto"):
    """
    Encode categorical features in a DataFrame using specified method.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    method (str): Encoding method ('onehot', 'label').
    
    Returns:
    pd.DataFrame: DataFrame with categorical features encoded.
    """
    df_encoded = df.copy()
    # Check if there are any categorical columns
    categorical_cols = df_encoded.select_dtypes(include=["object", "category"]).columns
    # Determine cardinality threshold for one-hot encoding
    max_unique = df.select_dtypes(include=["object", "category"]).nunique().max()
    if max_unique <= 10:
        cardinality_threshold = 10
    elif max_unique <= 50:
        cardinality_threshold = 20
    else:
        cardinality_threshold = 10
    # Encode categorical columns
    for col in categorical_cols:
        nunique = df_encoded[col].nunique()
        if method == "onehot" or (method == "auto" and nunique <= cardinality_threshold):
            # One-Hot Encode
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = df_encoded.drop(columns=col).join(dummies)
        elif method == "label" or (method == "auto" and nunique > cardinality_threshold):
            # Label Encode
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded
    
def scale_features(df, model="linear", method="auto"):
    """
    Scale features in a DataFrame using specified method.
    Method determined based on outliers & Regression-Classification task.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    model (str): Model type ('linear', 'logistic', 'KNN', 'tree').
    
    Returns:
    pd.DataFrame: DataFrame with features scaled.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_scaled = df.copy()
    # Check method
    if method == "auto":
        # Check model type
        if model in ["linear", "logistic", "tree"]:
            scaler = StandardScaler()
        elif model == "KNN":
            scaler=MinMaxScaler()
        else:
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
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    # Scale only numeric columns
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled
    
def sample_data(df, features, target, method="auto"):
    """
    Sample a specified number of rows from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to sample from.
    features (pd.DataFrame): Features DataFrame.
    target (pd.Series): Target Series.
    method (str): Sampling method ('auto', 'oversample', 'undersample', 'smote').
    
    Returns:
    new_features (pd.DataFrame): Sampled features DataFrame.
    new_target (pd.Series): Sampled target Series.
    """
    if method == 'auto':
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
    elif method == 'oversample':
        # Oversampling
        over_sampler = RandomOverSampler(sampling_strategy="minority", random_state=42)
        x, y = over_sampler.fit_resample(features, target)
    elif method == 'undersample':
        # Undersampling
        under_sampler = RandomUnderSampler(sampling_strategy="majority", random_state=42)
        x, y = under_sampler.fit_resample(features, target)
    elif method == 'smote':
        # SMOTE
        smote = SMOTE(sampling_strategy="minority", random_state=42)
        x, y = smote.fit_resample(features, target)
    return x, y

def preprocess_data(df, target, model="linear", value=0, fill_method="auto", encode_method="auto", scale_method="auto", sample_method="auto"):
    """
    Preprocess the DataFrame by filling missing values, encoding, scaling, and optional sampling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    target (str): Name of the target column.
    model (str): Model type ('linear', 'logistic', 'KNN', 'tree').
    value (any): Value to use for filling missing values if fill_method is 'constant'.
    fill_method (str): Missing value strategy.
    encode_method (str): Categorical encoding strategy.
    scale_method (str): Scaling strategy.
    sample_method (str): Sampling strategy.
    
    Returns:
    x (pd.DataFrame): Preprocessed feature matrix.
    y (pd.Series or None): Preprocessed target, or None if not provided.
    """
    df_cleaned=df.copy()
    y=None
    # Split features and target
    y = df_cleaned[target]
    x = df_cleaned.drop(columns=[target])
    # Clean and preprocess the data
    if fill_method == "constant":
        x = fill_missing_values(x, value, method=fill_method)
    else:
        x = fill_missing_values(x, method=fill_method)
    x = encode_categorical_features(x, method=encode_method)
    if scale_method == "auto":
        x = scale_features(x, model=model, method=scale_method)
    else:
        x = scale_features(x, method=scale_method)
    if y is not None:
        x, y = sample_data(pd.concat([x, y], axis=1), x, y, method=sample_method)
    return x, y
