import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
st.title("Machine learning application")
#input & handle data
st.header("User input")
file= st.file_uploader("Upload a CSV file", type=["csv"])
if st.button("Read"):
    if file is not None:
        # Read the CSV file
        try:
            data = pd.read_csv(file)
            st.write("Data loaded successfully!")
            st.write("Data Preview:", data)
        except Exception as e:
            st.error(f"Error loading data: {e}")
    else:
        st.error("Please upload a CSV file.")
