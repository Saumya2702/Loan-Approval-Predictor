import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('loan_model.joblib')
scaler = joblib.load('scaler.joblib')

def main():
    # Sidebar for instructions
    st.sidebar.markdown("## 📌 Instructions")
    st.sidebar.markdown(
        "- Fill in all fields accurately.\n"
        "- Loan approval is based on historical data patterns.\n"
        "- Higher income & credit score improve approval chances.\n"
        "- Click **Predict Loan Approval** to see the result."
    )
    
    st.markdown("<h1 style='text-align: center; color: #0044cc;'>Loan Approval Predictor</h1>", unsafe_allow_html=True)
    st.image("loan.jpg", width=500)
    st.markdown("<p style='text-align: center;'>Check if your loan will be approved based on financial data</p>", unsafe_allow_html=True)
    
    # Personal Information Section
    st.subheader("👤 Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("📅 Age", min_value=0, max_value=100, value=30)
        person_gender = st.selectbox("⚧ Gender", options=['Male', 'Female'])
        person_education = st.selectbox("🎓 Education Level", options=['High School', 'Bachelor', 'Master', 'Associate', 'Doctorate'])
        person_income = st.number_input("💰 Annual Income ($)", min_value=0, value=50000)
        person_emp_exp = st.number_input("⌛ Employment Experience (Years)", min_value=0, value=5)
        person_home_ownership = st.selectbox("🏡 Home Ownership", options=['Rent', 'Own', 'Mortgage', 'Other'])
    
    # Loan Information Section
    st.subheader("💳 Loan Information")
    with col2:
        loan_amnt = st.number_input("💲 Loan Amount ($)", min_value=0, value=10000)
        loan_intent = st.selectbox("🎯 Loan Purpose", options=['Personal', 'Education', 'Medical', 'Venture', 'Home Improvement', 'Debt Consolidation'])
        loan_int_rate = st.slider("% Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0)
        loan_percent_income = st.slider("💸 Loan as % of Income", min_value=0.0, max_value=1.0, value=0.2)
        cb_person_cred_hist_length = st.number_input("📅 Credit History Length (Years)", min_value=0, value=5)
        credit_score = st.slider("📊 Credit Score", min_value=300, max_value=850, value=600)
        previous_loan_defaults_on_file = st.selectbox("❌ Previous Loan Defaults?", options=['Yes', 'No'])
    
    # Convert categorical inputs to numerical values
    mappings = {
        'person_gender': {'Male': 1, 'Female': 0},
        'person_education': {'High School': 0, 'Bachelor': 1, 'Master': 2, 'Associate': 3, 'Doctorate': 4},
        'person_home_ownership': {'Rent': 0, 'Own': 1, 'Mortgage': 2, 'Other': 3},
        'loan_intent': {'Personal': 0, 'Education': 1, 'Medical': 2, 'Venture': 3, 'Home Improvement': 4, 'Debt Consolidation': 5},
        'previous_loan_defaults_on_file': {'Yes': 1, 'No': 0}
    }
    
    person_gender = mappings['person_gender'][person_gender]
    person_education = mappings['person_education'][person_education]
    person_home_ownership = mappings['person_home_ownership'][person_home_ownership]
    loan_intent = mappings['loan_intent'][loan_intent]
    previous_loan_defaults_on_file = mappings['previous_loan_defaults_on_file'][previous_loan_defaults_on_file]
    
    # Create a DataFrame
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
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
    })
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Prediction button
    if st.button("🚀 Predict Loan Approval"):
        prediction = model.predict(input_scaled)
        if prediction[0] == 1:
            st.success("✅ Congratulations! Your loan is approved.")
        else:
            st.error("❌ Unfortunately, your loan is not approved.")

if __name__ == '__main__':
    main()