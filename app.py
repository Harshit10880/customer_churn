import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("churn_svm_pipeline.joblib")

st.set_page_config(page_title="Customer Churn Prediction")

st.title("📊 Customer Churn Prediction")

# Inputs
gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=600.0
)

if st.button("Predict Churn"):

    gender_num = 1 if gender == "Male" else 0

    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract],
        "gender": [gender_num],
        "SeniorCitizen": [senior]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

# import joblib

# model = joblib.load("churn_svm_pipeline.joblib")
# print(type(model))