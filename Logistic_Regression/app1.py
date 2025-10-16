import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie
import requests

# Load animation (Lottie JSON)
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Diabetes animation
# ---- Header with Animation ----
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            st.error(f"❌ Couldn't load animation. Status code: {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        return None

# ✅ Use a valid Lottie animation URL
lottie_diabetes = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")

st.title("🩺 Diabetes Prediction App")

if lottie_diabetes:
    st_lottie(lottie_diabetes, height=250, key="diabetes")
else:
    st.warning("⚠️ Animation could not load. Check your URL or internet connection.")


# ---- Page Config ----
st.set_page_config(page_title="Diabetes Prediction", page_icon="💉", layout="centered")

# ---- Custom Background ----
st.markdown(
    """
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
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2e8b57;
        transform: scale(1.05);
    }
    .main-card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Header with Animation ----
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("<h1 style='color:#003366;'>💉 Diabetes Prediction App</h1>", unsafe_allow_html=True)
    st.write("Enter patient details below and predict the diabetes condition instantly.")
with col2:
    st_lottie(lottie_diabetes, height=180, key="diabetes_anim2")

# ---- Form Section ----
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.subheader("🧍‍♀️ Enter Patient Data")
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=850, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=25.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

st.markdown('</div>', unsafe_allow_html=True)

# ---- Predict Button ----
submit_button = st.button("🔍 Predict")

if submit_button:
    # Prepare input
    user_input = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                           bmi, diabetes_pedigree_function, age]).reshape(1, -1)

    # Dummy scaler (replace with real one if available)
    dummy_data = np.random.rand(10, 8)
    scaler = StandardScaler()
    scaler.fit(dummy_data)
    scaled_input = scaler.transform(user_input)

    # Dummy prediction model (replace with real one)
    class DummyModel:
        def predict(self, x):
            return [1 if x[0][1] > 120 else 0]
    regression = DummyModel()

    prediction = regression.predict(scaled_input)

    # ---- Result Section ----
    if prediction[0] == 1:
        st.success("🩸 The model predicts: **Diabetic**")
        st.balloons()
    else:
        st.info("✅ The model predicts: **Non-Diabetic**")

