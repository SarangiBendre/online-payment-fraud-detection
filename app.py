import streamlit as st
import pickle
import pandas as pd

# ===============================
# Load model, threshold & features
# ===============================
with open("models/fraud_model_final.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
threshold = data["threshold"]
feature_names = data["features"]   # VERY IMPORTANT

# ===============================
# Page configuration
# ===============================
st.set_page_config(
    page_title="Online Payment Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Online Payment Fraud Detection System")
st.write(
    "This system detects **potentially fraudulent online payment transactions** "
    "using a Machine Learning model trained on real-world transaction patterns."
)

st.markdown("---")

# ===============================
# Input section
# ===============================
st.subheader("🔍 Enter Transaction Details")

amount = st.number_input("Transaction Amount", min_value=0.0, step=100.0)

oldbalanceOrg = st.number_input("Sender Balance Before Transaction")
newbalanceOrig = st.number_input("Sender Balance After Transaction")

oldbalanceDest = st.number_input("Receiver Balance Before Transaction")
newbalanceDest = st.number_input("Receiver Balance After Transaction")

transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"]
)

st.markdown("---")

# ===============================
# Prediction logic
# ===============================
if st.button("🚨 Check Fraud Risk"):

    # Base numerical features
    input_dict = {
    "step": 1,   # 👈 auto-filled system value
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "balanceDiffOrig": oldbalanceOrg - newbalanceOrig,
    "balanceDiffDest": newbalanceDest - oldbalanceDest
}


    # Create DataFrame (1 row)
    input_df = pd.DataFrame([input_dict])

    # Add missing transaction-type columns
    for col in feature_names:
        if col.startswith("type_") and col not in input_df.columns:
            input_df[col] = 0

    # Set selected transaction type
    input_df[f"type_{transaction_type}"] = 1

    # Reorder columns EXACTLY like training
    input_df = input_df[feature_names]

    # Prediction
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if probability >= threshold:
        st.error(f"⚠ Fraud Detected")
        st.warning(f"Risk Score: **{probability:.2f}**")
        st.write("This transaction shows strong patterns associated with fraudulent behavior.")
    else:
        st.success(f"✅ Legitimate Transaction")
        st.info(f"Risk Score: **{probability:.2f}**")
        st.write("This transaction appears safe based on learned patterns.")

st.markdown("---")
st.caption("⚙️ Powered by XGBoost & Machine Learning")
