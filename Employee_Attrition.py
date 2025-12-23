import streamlit as st
import pandas as pd
import joblib


# Page Configuration

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Employee Attrition Prediction System")
st.write("Predict whether an employee is likely to leave the organization.")


# Load Model & Artifacts

@st.cache_resource
def load_artifacts():
    model = joblib.load("attrition_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()


# Input Section

st.subheader("🧑 Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=60, value=32)
    job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=6)
    overtime = st.selectbox("OverTime", ["No", "Yes"])

with col2:
    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=200000,
        value=45000,
        step=1000
    )
    work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])


# Encoding (MATCHES TRAINING)

overtime = 1 if overtime == "Yes" else 0
gender = 1 if gender == "Male" else 0

marital_status_map = {
    "Divorced": 0,
    "Married": 1,
    "Single": 2
}
marital_status = marital_status_map[marital_status]


# Create Input DataFrame

input_data = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [monthly_income],
    "JobSatisfaction": [job_satisfaction],
    "WorkLifeBalance": [work_life_balance],
    "YearsAtCompany": [years_at_company],
    "OverTime": [overtime],
    "Gender": [gender],
    "MaritalStatus": [marital_status]
})

# Align with training feature order

input_data = input_data.reindex(columns=feature_names, fill_value=0)


# Prediction

if st.button("🔍 Predict Attrition"):

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    # IMPORTANT:
    # Class 0 = Leave
    # Class 1 = Stay
    leave_probability = model.predict_proba(input_scaled)[0][0]

    st.subheader("📊 Prediction Result")

    if prediction == 0:
        st.error("❌ YES – Employee is likely to leave")
    else:
        st.success("✅ NO – Employee is likely to stay")

    st.write(f"**Attrition Probability:** {leave_probability:.2f}")


# Footer

st.markdown("---")
st.caption("Employee Attrition Prediction")