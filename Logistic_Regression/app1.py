import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import time

# ---- Load saved scaler and model ----
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
model_path = os.path.join(os.path.dirname(__file__), "Logistic_Regression_model.pkl")

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(model_path, "rb") as f:
    regression = pickle.load(f)

# ---- Page Config ----
st.set_page_config(page_title="Diabetes Prediction", page_icon="💉", layout="wide")

# ---- Custom CSS ----
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #89f7fe, #66a6ff);
    font-family: "Poppins", sans-serif;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    font-size: 1em;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    background-color: #2e8b57;
    transform: scale(1.08);
}
.main-card {
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 6px 25px rgba(0,0,0,0.25);
    margin-top: 20px;
    transition: all 0.3s ease-in-out;
}
.main-card:hover {
    transform: translateY(-5px);
}
@media (max-width: 768px) {
    .main-card {
        padding: 20px;
        margin-top: 10px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details below and predict the diabetes condition instantly.")

# ---- Animated Input Function (optional) ----
def animated_number_input(label, value, min_value, max_value, step=1):
    container = st.empty()
    for v in range(min_value, value+1, step):
        container.number_input(label, value=v, min_value=min_value, max_value=max_value)
        time.sleep(0.01)
    return container.number_input(label, value=value, min_value=min_value, max_value=max_value)

# ---- Input Form ----
st.markdown('<div class="main-card">', unsafe_allow_html=True)
cols = st.columns(2)

with cols[0]:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=20)

with cols[1]:
    insulin = st.number_input("Insulin", min_value=0, max_value=850, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=25.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

st.markdown('</div>', unsafe_allow_html=True)

# ---- Predict Button ----
submit_button = st.button("🔍 Predict")

if submit_button:
    user_input = np.array([pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)
    
    scaled_input = scaler.transform(user_input)
    prediction = regression.predict(scaled_input)
    
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    if prediction[0] == 1:
        st.success("🩸 The model predicts: **Diabetic**")
    else:
        st.info("✅ The model predicts: **Non-Diabetic**")
    
    st.markdown('</div>', unsafe_allow_html=True)


