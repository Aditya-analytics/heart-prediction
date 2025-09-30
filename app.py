import streamlit as st
import pandas as pd
import joblib

# Load saved pipeline (includes preprocessing + model)
model_pipeline = joblib.load('ML_Projects\heart_disease_pred\heart-prediction\model_pipeline.pkl')



# Title
st.title("❤️ Heart Disease Risk Prediction")
st.subheader("Provide the following details:")

# Age slider at top
age = st.slider("Age", min_value=1, max_value=120, value=30, step=1)

# Layout: two columns for inputs
left, right = st.columns(2)

with left:
    resting_bp = st.number_input("RestingBP", min_value=0, max_value=200, value=120, step=1)
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200, step=1)
    fasting_bs = st.selectbox("FastingBS (Blood Sugar > 120 mg/dl)", ["No (0)", "Yes (1)"])
    exercise_angina = st.selectbox("Exercise Angina", ["Yes", "No"])
    st_slope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])

with right:
    max_hr = st.number_input("MaxHR", min_value=60, max_value=220, value=150, step=1)
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

# Convert inputs
fasting_bs = 1 if "Yes" in fasting_bs else 0
exercise_angina = 1 if exercise_angina == "Yes" else 0
sex = 1 if sex == "Male" else 0   # example: 1=Male, 0=Female

# Center the Predict button
l, m, r = st.columns([1, 2, 1])
with m:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

# Prediction logic
if predict_btn:
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

    # Predict using pipeline
    prediction = model_pipeline.predict(input_data)[0]
    prob = model_pipeline.predict_proba(input_data)[0][1]

    # Show result
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Probability: {prob:.2f})")
