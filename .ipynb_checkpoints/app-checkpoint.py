import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Load Model
# ==========================================

model = joblib.load("churn_prediction_model.pkl")


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# Title
# ==========================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn "
    "based on their profile and subscription information."
)


st.divider()


# ==========================================
# Customer Information
# ==========================================

st.subheader("Customer Information")


col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )


with col2:

    contract_type = st.selectbox(
        "Contract Type",
        [
            "Month-to-Month",
            "One-Year",
            "Two-Year"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber Optic",
            "No Internet"
        ]
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=800.0,
        step=10.0
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )


st.divider()


# ==========================================
# Prediction
# ==========================================

if st.button(
    "Predict Churn",
    type="primary",
    use_container_width=True
):

    # Create DataFrame
    customer_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "ContractType": [contract_type],
        "InternetService": [
            None if internet_service == "No Internet"
            else internet_service
        ],
        "TotalCharges": [total_charges],
        "TechSupport": [tech_support]
    })


    # Prediction probability
    probability = model.predict_proba(
        customer_data
    )[0][1]


    # Prediction
    prediction = int(probability >= 0.5)


    # Risk level
    if probability < 0.40:
        risk = "Low"
    elif probability < 0.70:
        risk = "Medium"
    else:
        risk = "High"


    # ======================================
    # Results
    # ======================================

    st.subheader("Prediction Result")


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )


    with result_col2:

        st.metric(
            "Risk Level",
            risk
        )


    with result_col3:

        if prediction == 1:

            st.metric(
                "Prediction",
                "Likely to Churn"
            )

        else:

            st.metric(
                "Prediction",
                "Likely to Stay"
            )


    # ======================================
    # Recommendation
    # ======================================

    st.divider()

    if risk == "High":

        st.error(
            "⚠️ High churn risk. "
            "Consider offering a personalized retention "
            "offer or contacting the customer."
        )

    elif risk == "Medium":

        st.warning(
            "⚠️ Medium churn risk. "
            "Monitor this customer and consider targeted "
            "engagement."
        )

    else:

        st.success(
            "✅ Low churn risk. "
            "Customer currently appears relatively stable."
        )