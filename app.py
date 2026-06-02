import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_svm_pipeline.joblib")
# or pickle.load(...) if you saved as pkl locally

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=600.0
)

if st.button("Predict"):

    gender_num = 1 if gender == "Male" else 0

    input_df = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total],
        "Contract": [contract],
        "gender": [gender_num],
        "SeniorCitizen": [senior]
    })

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")