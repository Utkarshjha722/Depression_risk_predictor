import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the model
with open('depression_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load the dataset (only for column information)
dataset = pd.read_csv('mental_health_and_technology_usage_2024.csv')
dataset = dataset.drop(['Online_Support_Usage', 'Support_Systems_Access', 'User_ID'], axis=1)

# Separate features and target variable (only for column information)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1].values

# Identify categorical features
categorical_features = ['Gender', 'Mental_Health_Status', 'Stress_Level']

# Create ColumnTransformer (using loaded model's configuration)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features),
                                     ('scaler', StandardScaler(), [col for col in X.columns if col not in categorical_features])],
                       remainder='passthrough')

ct.fit(X)  # Fit the ColumnTransformer with the training data

# Streamlit UI
st.title("Depression Risk Prediction")

# User input
gender = st.selectbox("Gender", options=['Male', 'Female', 'Other'])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
daily_usage_hours = st.number_input("Daily Usage Hours", min_value=0.0, max_value=24.0, value=8.0)
mental_health_status = st.selectbox("Mental Health Status", options=['Stable', 'Unstable'])
stress_level = st.selectbox("Stress Level", options=['Low', 'Medium', 'High'])
social_media_usage = st.number_input("Social Media Usage", min_value=0.0, max_value=24.0, value=2.0)
technology_usage_hours = st.number_input("Technology Usage Hours", min_value=0.0, max_value=24.0, value=4.0)
social_media_usage_hours = st.number_input("Social Media Usage Hours", min_value=0.0, max_value=24.0, value=2.0)
gaming_hours = st.number_input("Gaming Hours", min_value=0.0, max_value=24.0, value=1.0)
screen_time_hours = st.number_input("Screen Time Hours", min_value=0.0, max_value=24.0, value=6.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
physical_activity_hours = st.number_input("Physical Activity Hours", min_value=0.0, max_value=24.0, value=1.0)

# Create user data dictionary
user_data = {
    'Gender': gender,
    'Age': age,
    'Daily_Usage_Hours': daily_usage_hours,
    'Mental_Health_Status': mental_health_status,
    'Stress_Level': stress_level,
    'Social_Media_Usage': social_media_usage,
    'Technology_Usage_Hours': technology_usage_hours,
    'Social_Media_Usage_Hours': social_media_usage_hours,
    'Gaming_Hours': gaming_hours,
    'Screen_Time_Hours': screen_time_hours,
    'Sleep_Hours': sleep_hours,
    'Physical_Activity_Hours': physical_activity_hours
}

# Function to predict Mental_Health_Status
def predict_mental_health_status(user_data):
    user_data_df = pd.DataFrame([user_data])
    user_data_df = user_data_df[X.columns]
    user_data_transformed = ct.transform(user_data_df)
    prediction = classifier.predict(user_data_transformed)[0]
    return prediction

# Prediction
if st.button("Predict"):
    prediction = predict_mental_health_status(user_data)
    st.write(f"Predicted Mental Health Status: {prediction}")