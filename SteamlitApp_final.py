#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests
import json
import xgboost
import os
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from google.cloud import storage
from google.oauth2 import service_account


# In[2]:


# Set the path to your service account key file
key_file_path = r'C:\Users\002736125\Desktop\PERSONAL\School\Analytics\ALY 6980 CAPSTONE\healthy-booth-427815-g9-4d1f941d26d0.json'

# Set the environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file_path


# In[3]:


credentials = service_account.Credentials.from_service_account_file(r'C:\Users\002736125\Desktop\PERSONAL\School\Analytics\ALY 6980 CAPSTONE\healthy-booth-427815-g9-4d1f941d26d0.json')


# In[4]:


# Initialize Google Cloud Storage client
client = storage.Client(credentials=credentials)

# Replace 'your_bucket_name' with your Google Cloud Storage bucket name
bucket_name = 'aly6980'
blob_name = 'New_MDS.csv'

# Get the bucket and blob (file)
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_name)

# Download the file as bytes
file_content = blob.download_as_string()

# Read CSV from bytes
df = pd.read_csv(io.BytesIO(file_content))


# In[5]:


import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Selecting the specified variables
variables_to_use = [
    'PP Buy Box', 'PP Units', 'PP Sales', 'PP Orders', 'PP CR%',
    'PY Page Views', 'PY Buy Box', 'PY Units', 'PY Sales', 'PY Orders', 'PY CR%',
    'Total Sales', 'Total Campaigns', 'Ad Spend'
]

# Dropping rows with missing values and selecting features and target variable
df_cleaned = df[variables_to_use].dropna()
X = df_cleaned.drop(columns=['Total Sales'])  # Features
y = df_cleaned['Total Sales']  # Target variable

# Instantiate the Gradient Boosting Regressor model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
gb_model.fit(X, y)

# Define the Streamlit app
def main():
    st.title("Total Sales Prediction App")
    
    # Add input fields for variables
    pp_buy_box = st.number_input("Enter PP Buy Box", value=1)
    pp_units = st.number_input("Enter PP Units", value=10)
    pp_sales = st.number_input("Enter PP Sales", value=385)
    pp_orders = st.number_input("Enter PP Orders", value=9)
    pp_cr = st.number_input("Enter PP CR%", value=0.2)
    py_page_views = st.number_input("Enter PY Page Views", value=289)
    py_buy_box = st.number_input("Enter PY Buy Box", value=1)
    py_units = st.number_input("Enter PY Units", value=10)
    py_sales = st.number_input("Enter PY Sales", value=405)
    py_orders = st.number_input("Enter PY Orders", value=9)
    py_cr = st.number_input("Enter PY CR%", value=0.2)
    total_campaigns = st.number_input("Enter Total Campaigns", value=162)
    ad_spend = st.number_input("Enter Ad Spend", value=39)
    # Add input fields for other variables...
    
    # Make predictions
    if st.button("Predict Total Sales"):
        input_data = [[pp_buy_box, pp_units, pp_sales, pp_orders, pp_cr, py_page_views, py_buy_box,
                       py_units, py_sales, py_orders, py_cr, total_campaigns, ad_spend]]
        prediction = gb_model.predict(input_data)
        st.success(f"Predicted Total Sales: {prediction[0]}")

if __name__ == "__main__":
    main()


# In[ ]:




