import streamlit as st
import pandas as pd
import joblib

# Load saved pipeline (includes preprocessing + model)
model_pipeline = joblib.load(r"C:\Users\suyas\python.py\ML_Projects\heart_disease_pred\model_pipeline.pkl")

st.title("❤️ Heart Disease Risk Prediction")

st.subheader("Provide the following details:")

# Numeric inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
resting_bp = st.number_input("RestingBP", min_value=0, max_value=200, value=120, step=1)
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200, step=1)
fasting_bs = st.radio("FastingBS (Blood Sugar > 120 mg/dl)", [0, 1])
max_hr = st.number_input("MaxHR", min_value=60, max_value=220, value=150, step=1)
oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Categorical inputs (raw values, not encoded)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
exercise_angina = st.selectbox("Exercise Angina", ["Yes", "No"])
st_slope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])

# Prediction button
if st.button("Predict"):
    # Create input DataFrame (raw features)
    input_data = pd.DataFrame([{
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingECG": resting_ecg,
        "ExerciseAngina": exercise_angina,
        "ST_Slope": st_slope
    }])

    # Predict using pipeline (handles encoding + scaling)
    prediction = model_pipeline.predict(input_data)[0]
    prob = model_pipeline.predict_proba(input_data)[0][1]  # probability of heart disease

    # Show result
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Probability: {prob:.2f})")
