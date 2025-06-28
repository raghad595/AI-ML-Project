from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
def evaluate_model(model, x_test, y_test, model_type="classification"):
    """
    Evaluate the performance of a trained machine learning model.
    
    Parameters:
    model: Trained machine learning model.
    x_test (pd.DataFrame): Test feature matrix.
    y_test (pd.Series or np.array): True labels for the test set.
    model_type (str): Type of model ('classification' or 'regression').
    
    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    
    predictions = model.predict(x_test)
    
    if model_type == "classification":
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report
        }
    
    elif model_type == "regression":
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)
        
        return {
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "r2_score": r2
        }
    
