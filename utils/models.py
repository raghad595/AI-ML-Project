from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

def train_model(df, features, target, test, model_type="linear"):
    """
    Train a machine learning model on the provided DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    features (list): List of feature column names.
    target (str): Name of the target column.
    model_type (str): Type of model to train ('linear', 'logistic', 'KNN', 'tree').
    
    Returns:
    model: Trained machine learning model.
    """
    # Preprocessed data
    x, y = df[features].copy(), df[target].copy()
    # Train the model
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test, random_state=42)
    # Classification models
    if model_type == "logistic":
        model = LogisticRegression()
    elif model_type == "KNN":
        model = KNeighborsClassifier()
    elif model_type == "svm":
        model = SVC()
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=10)
    elif model_type == "tree":
        model = DecisionTreeClassifier()
    elif model_type == "xgboost":
        model = XGBClassifier()
    # Regression models
    elif model_type == "linear":
        model = LinearRegression()
    elif model_type == "KNN_regressor":
        model = KNeighborsRegressor()
    elif model_type == "svm_regressor":
        model = SVR()
    elif model_type == "random_forest_regressor":
        model = RandomForestRegressor()
    elif model_type == "DT_regressor":
        model = DecisionTreeRegressor()
    elif model_type == "xgboost_regressor":
        model = XGBRegressor()
    model.fit(xtrain, ytrain)
    # Predict on the test set
    predictions = model.predict(xtest)
    
    
    return model, predictions, xtrain, xtest, ytrain, ytest
