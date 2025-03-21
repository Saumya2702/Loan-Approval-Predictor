import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit Page Configuration
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="wide")

# Title with styling
st.markdown(
    "<h1 style='text-align: center; color: #007bff;'>Loan Approval Predictor</h1>",
    unsafe_allow_html=True
)

# Add a stylish header image
st.image("loan.jpg", width=500)

st.markdown("<h5 style='text-align: center;'>Check if your loan will be approved based on financial data</h5>", unsafe_allow_html=True)

# Sidebar for Instructions
with st.sidebar:
    st.header("📌 Instructions")
    st.write("""
    - Fill in all the fields accurately.
    - Your loan approval is based on historical data patterns.
    - Higher income & credit score improve approval chances.
    - Click 'Predict Loan Approval' to see the result.
    """)

# UI Layout - Two Columns
col1, col2 = st.columns([1, 1])

# --- Column 1: Personal Details ---
with col1:
    st.subheader("👤 Personal Information")
    
    person_age = st.number_input("📅 Age", min_value=18, max_value=100, value=30)
    person_gender = st.selectbox("⚧ Gender", ['Male', 'Female'])
    person_education = st.selectbox("🎓 Education Level", ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
    person_income = st.number_input("💵 Annual Income ($)", min_value=0, step=1000, value=50000)
    person_emp_exp = st.number_input("👔 Employment Experience (Years)", min_value=0, step=1, value=5)
    person_home_ownership = st.selectbox("🏠 Home Ownership", ['Rent', 'Own', 'Mortgage', 'Other'])

# --- Column 2: Loan Details ---
with col2:
    st.subheader("💳 Loan Information")

    loan_amnt = st.number_input("🏦 Loan Amount ($)", min_value=0, step=1000, value=20000)
    loan_intent = st.selectbox("📌 Loan Purpose", ['Personal', 'Education', 'Medical', 'Venture', 'Home Improvement', 'Debt Consolidation'])
    loan_int_rate = st.slider("📉 Interest Rate (%)", min_value=0.0, max_value=40.0, step=0.1, value=5.0)
    loan_percent_income = st.slider("💸 Loan as % of Income", min_value=0.0, max_value=100.0, step=0.1, value=20.0)
    cb_person_cred_hist_length = st.number_input("📜 Credit History Length (Years)", min_value=0, step=1, value=5)
    credit_score = st.slider("🏆 Credit Score", min_value=300, max_value=850, step=10, value=700)
    previous_loan_defaults_on_file = st.selectbox("❌ Previous Loan Defaults?", ['No', 'Yes'])

# Convert categorical inputs to numeric
person_gender_map = {'Male': 0, 'Female': 1}
person_education_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
person_home_ownership_map = {'Rent': 0, 'Own': 1, 'Mortgage': 2, 'Other': 3}
loan_intent_map = {'Personal': 0, 'Education': 1, 'Medical': 2, 'Venture': 3, 'Home Improvement': 4, 'Debt Consolidation': 5}
previous_loan_defaults_map = {'No': 0, 'Yes': 1}

# Apply mapping
person_gender = person_gender_map[person_gender]
person_education = person_education_map[person_education]
person_home_ownership = person_home_ownership_map[person_home_ownership]
loan_intent = loan_intent_map[loan_intent]
previous_loan_defaults_on_file = previous_loan_defaults_map[previous_loan_defaults_on_file]

# Prediction Function
def make_prediction():
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_gender': [person_gender],
        'person_education': [person_education],
        'person_income': [person_income],
        'person_emp_exp': [person_emp_exp],
        'person_home_ownership': [person_home_ownership],
        'loan_amnt': [loan_amnt],
        'loan_intent': [loan_intent],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income / 100],  # Convert % to decimal
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
    })

    # Scale input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    return prediction[0]

# Prediction Button with Animation
if st.button("💡 Predict Loan Approval", key="predict_button"):
    with st.spinner("🔍 Analyzing Your Data..."):
        prediction = make_prediction()
        if prediction == 1:
            st.success("✅ **Congratulations! Your loan is approved.**")
        else:
            st.error("❌ **Unfortunately, your loan is not approved.**")
