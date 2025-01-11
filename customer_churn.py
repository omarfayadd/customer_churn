#nm
import streamlit as st 
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Bank Customer Churn Prediction App", layout="centered", page_icon="📊")

st.title('Bank Customer Churn Prediction App')
st.image('Customer Churn.jpg')
st.text('This application can predict whether the customer will leave the bank or not')
st.text('Please fill-in the following Data :')

CreditScore = st.number_input("CreditScore")
Age = st.slider("Age",18,62)
Tenure = st.slider("Tenure",0,10)
Balance = st.number_input("Balance")
NumOfProducts = st.slider("NumOfProducts",1,4)
HasCrCard = st.slider("HasCrCard",0,1)
IsActiveMember = st.slider("IsActiveMember",0,1)
EstimatedSalary = st.number_input("EstimatedSalary")
Geography = st.radio("Geography_encoded" , ["France","Germany","Spain"])
Gender = st.selectbox("Gender" , ["Male" , "Female"])
button = st.button("Predict")

if button == True: 
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('model.pkl')

    Geography_mapping = {'France': 0, 'Germany': 1 , 'Spain' :2}
    Gender_mapping = {'Male': 1, 'Female': 0}
    
    
    Geography_encoded = Geography_mapping[Geography]
    Gender_encoded = Gender_mapping[Gender]
    
    input_data = np.array([[CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard , IsActiveMember ,EstimatedSalary , Geography_encoded , Gender_encoded ]])
    input_data_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_data_scaled)
    if prediction == 1:
        st.success("The customer is leave the bank.")
    else:
        st.success("The customer is stay with the bank.")