import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('loan_model.pkl')

# Page Configuration
st.set_page_config(page_title="Loan Approval System", layout="wide")

st.title("Loan Approval Prediction System 💰")
st.write("Enter client details below to predict the loan status based on our ML model.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income", min_value=0, value=50000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=15000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)

with col2:
    # Gender: Female=0, Male=1 (alphabetical)
    gender = st.selectbox("Gender", options=["Female", "Male"])
    gender_val = 0 if gender == "Female" else 1

    # Education: Bachelors=0, High School=1, Masters=2, PhD=3
    education = st.selectbox("Education Level", options=["Bachelors", "High School", "Masters", "PhD"])
    education_dict = {"Bachelors": 0, "High School": 1, "Masters": 2, "PhD": 3}
    education_val = education_dict[education]

    # City: Chicago=0, Houston=1, New York=2, San Francisco=3
    city = st.selectbox("City", options=["Chicago", "Houston", "New York", "San Francisco"])
    city_dict = {"Chicago": 0, "Houston": 1, "New York": 2, "San Francisco": 3}
    city_val = city_dict[city]

    # EmploymentType: Salaried=0, Self-Employed=1, Unemployed=2
    emp_type = st.selectbox("Employment Type", options=["Salaried", "Self-Employed", "Unemployed"])
    emp_dict = {"Salaried": 0, "Self-Employed": 1, "Unemployed": 2}
    emp_val = emp_dict[emp_type]

st.divider()

if st.button("Predict Loan Status"):
    input_data = np.array([[age, income, loan_amount, credit_score, years_exp,
                            gender_val, education_val, city_val, emp_val]])
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("Result: The Loan is likely to be APPROVED! ✅")
            st.balloons()
        else:
            st.error("Result: The Loan is likely to be REJECTED. ❌")
    except Exception as e:
        st.error("An error occurred during prediction.")
        st.write("Error Details:", e)

st.caption("AI Final Project - Computer Science Department")
