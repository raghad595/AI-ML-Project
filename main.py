import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from utils.preprocessing import preprocess_data
from utils.models import train_model
from utils.evaluation import evaluate_model
st.title("Machine learning application")

#input & handle data
st.sidebar.header("User input")
file= st.file_uploader("Upload a CSV file", type=["csv"])
if st.sidebar.button("Read"):
    if file is not None:
        # Read the CSV file
        try:
            st.session_state["data"] = pd.read_csv(file)
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {e}")
    else:
        st.error("Please upload a CSV file then click read.")
    
#Data exploration
if "data" in st.session_state:
    data = st.session_state["data"]
    st.subheader("Data preview:")
    st.dataframe(data)
    #Data Exploration
    st.header("Data Exploration")
    options = ["Data Summary", "Data Types", "Missing Values", "Correlation Matrix"]
    option=st.sidebar.radio("Select an option:", options)
    if option == "Data Summary":
        with st.expander("Data Summary", expanded=True):
            st.write(data.describe())
    elif option == "Data Types":
        with st.expander("Data Types", expanded=True):
            st.write(data.dtypes)
    elif option == "Missing Values":
        with st.expander("Missing Values", expanded=True):
            st.write(data.isnull().sum())
    elif option == "Correlation Matrix":
        with st.expander("Correlation Matrix", expanded=True):
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                st.warning("No numeric columns found for correlation analysis.")
            else:
                correlation_matrix = numeric_data.corr()
                st.write(correlation_matrix)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
                
    # Select features & target variable
    st.sidebar.header("Feature Selection")
    features=st.sidebar.multiselect("Select features:", data.columns.tolist())
    target_col = [col for col in data.columns if col not in features]
    if target_col:
        target=st.sidebar.multiselect("Select target variable:", target_col)
    else:
        target = st.sidebar.multiselect("Select target variable:", data.columns.tolist(), index=0)
    if st.sidebar.button("Confirm"):
        if not features or not target:
            st.error("Please select at least one feature and one target variable.")
        else:
            st.session_state["features"] = features
            st.session_state["target"] = target
            st.success("Features and target variable selected successfully!")
            df_features = data[features]
            df_target = data[target]
            with st.expander("Selected Features and Target Variable", expanded=True):
                st.write("Features:")
                st.dataframe(df_features)
                st.write("Target Variable:")
                st.dataframe(df_target)
                
# Data preprocessing
if "features" in st.session_state and "target" in st.session_state:
    st.sidebar.header("Data Preprocessing")
    st.sidebar.subheader("Missing Value Handling")
    fill_method = st.sidebar.selectbox("Select missing value handling method:", ["auto", "mean", "median", "mode", "constant", "ffill", "bfill", "drop"])
    if fill_method == "constant":
        fill_value = st.sidebar.text_input("Enter constant value for missing values:")
    else:
        fill_value = None
    # Encode categorical variables
    encode_method = st.sidebar.selectbox("Select encoding method:", ["auto", "onehot", "label"])
    if encode_method == "auto":
        if data[st.session_state["features"]].select_dtypes(include=["object", "category"]).nunique().max() <= 10:
            encode_method = "onehot"
        else:
            encode_method = "label"
    # Scale numeric features
    scale_method = st.sidebar.selectbox("Select scaling method:", ["auto", "standard", "minmax"])
    # Select training options
    st.sidebar.header("Training Options")
    training_options = st.sidebar.radio("Select an option:", ["Classification", "Regression"])
    if training_options == "Classification":
        model_type = st.sidebar.selectbox("Select classification model type:", ["logistic", "KNN", "svm", "random_forest", "tree", "xgboost"])
    elif training_options == "Regression":
        model_type = st.sidebar.selectbox("Select regression model type:", ["linear", "KNN_regressor", "svm_regressor", "random_forest_regressor", "DT_regressor", "xgboost_regressor"])
    #Sample data
    sample_method = st.sidebar.selectbox("Select sampling method:", ["auto", "oversample", "undersample", "smote"])
    # Preprocess the data
    if st.sidebar.button("Preprocess Data"):
        try:
            x, y = preprocess_data(
                df=data,
                target=st.session_state["target"][0],
                fill_method=fill_method,
                encode_method=encode_method,
                scale_method=scale_method,
                sample_method=sample_method,
                value=fill_value
            )
            st.session_state["x"] = x
            st.session_state["y"] = y
            st.success("Data preprocessed successfully!")
            with st.expander("Preprocessed Features and Target Variable", expanded=True):
                st.subheader("Preprocessed Features")
                st.dataframe(x)
                st.subheader("Preprocessed Target")
                st.dataframe(y)
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")

# Model training
if "x" in st.session_state and "y" in st.session_state:
    st.header("Model Training")
    test_size = st.sidebar.slider("Select test size (fraction):", 0.1, 0.5, 0.2, 0.05)
    if st.button("Train Model"):
        try:
            model, predictions, xtrain, xtest, ytrain, ytest = train_model(
                df=pd.concat([st.session_state["x"], st.session_state["y"]], axis=1),
                features=st.session_state["features"],
                target=st.session_state["target"][0],
                test=test_size,
                model_type=model_type
            )
            st.session_state["model"] = model
            st.session_state["predictions"] = predictions
            st.success("Model trained successfully!")
            st.subheader("Model Predictions")
            st.dataframe(pd.DataFrame(predictions, columns=[st.session_state["target"][0]]))
        except Exception as e:
            st.error(f"Error during model training: {e}")
          
# Evaluation  
if "model" in st.session_state and "x" in st.session_state and "y" in st.session_state:
    st.sidebar.header("Model Evaluation")
    if st.sidebar.button("Evaluate Model"):
        try:
            evaluation_metrics = evaluate_model(
                model=st.session_state["model"],
                x_test=st.session_state["x"],
                y_test=st.session_state["y"],
                model_type=training_options.lower()  # Convert to lowercase for consistency
            )
            st.session_state["evaluation_metrics"] = evaluation_metrics
            st.success("Model evaluated successfully!")
            st.subheader("Evaluation Metrics")
            st.write(evaluation_metrics)
        except Exception as e:
            st.error(f"Error during model evaluation: {e}") 
