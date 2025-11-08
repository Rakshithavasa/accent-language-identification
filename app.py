import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
import pandas as pd
import datetime
import os
from io import BytesIO
from st_audiorec import st_audiorec  # 🎤 mic input

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Accent / Language Identification",
    page_icon="🎧",
    layout="centered",
)

# -----------------------------
# Custom Background & Style
# -----------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
    color: #000000;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
h1, h2, h3 {
    color: #003366;
    text-align: center;
    font-family: 'Poppins', sans-serif;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -----------------------------
# Load trained model and tools
# -----------------------------
try:
    model = joblib.load("rf_mfcc_model.joblib")
    scaler = joblib.load("scaler.joblib")
    encoder = joblib.load("label_encoder.joblib")
    st.success("✅ Model and encoders loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")

# -----------------------------
# Streamlit Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📁 Upload File", "🎤 Record Voice"])

audio_file = None
audio_source = None

# --- Upload Section ---
with tab1:
    uploaded_file = st.file_uploader("Upload a `.wav` file", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        audio_file = uploaded_file
        audio_source = "Uploaded"

# --- Recording Section ---
with tab2:
    st.write("Click below to record your voice 🎙️")
    recorded_audio = st_audiorec()
    if recorded_audio is not None:
        audio_file = BytesIO(recorded_audio)
        st.audio(audio_file, format="audio/wav")
        audio_source = "Recorded"

# -----------------------------
# Process the audio (upload or mic)
# -----------------------------
if audio_file is not None:
    audio_data, sr = sf.read(audio_file)

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    if len(audio_data) < sr * 0.5:
        st.warning("⚠️ Audio too short! Please record or upload at least 1 second.")
        st.stop()

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)

    try:
        features_scaled = scaler.transform(mfcc_mean)
        prediction = model.predict(features_scaled)
        predicted_label = encoder.inverse_transform(prediction)[0]

        confidence = (
            np.max(model.predict_proba(features_scaled)) * 100
            if hasattr(model, "predict_proba")
            else 0
        )

        # Log data
        log_data = {
            "timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "source": [audio_source],
            "predicted_language": [predicted_label],
            "confidence": [f"{confidence:.2f}"]
        }
        log_df = pd.DataFrame(log_data)
        if not os.path.exists("user_predictions.csv"):
            log_df.to_csv("user_predictions.csv", index=False)
        else:
            log_df.to_csv("user_predictions.csv", mode="a", header=False, index=False)

        # Display result
        st.markdown(
            f"""
            <div style="
                background-color: #ffffffcc;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                margin-top: 20px;
                text-align: center;
            ">
                <h3 style="color:#003366;">🎯 Predicted Accent/Language</h3>
                <h2 style="color:#007bff;">{predicted_label}</h2>
                <p style="font-size:18px;">Confidence: <b>{confidence:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 🍴 Cuisine Recommendation Section
        st.markdown("### 🍴 Accent-Aware Cuisine Recommendation")

        cuisine_dict = {
            "English": ["Grilled Chicken", "Tacos", "Pizza", "Burger"],
            "Tamil": ["Dosa", "Idli", "Sambar", "Rasam"],
            "Hindi": ["Butter Chicken", "Paneer Tikka", "Dal Makhani"],
            "Telugu": ["Pesarattu", "Pulihora", "Gutti Vankaya Curry"],
            "Kannada": ["Ragi Mudde", "Bisi Bele Bath", "Neer Dosa"],
            "Malayalam": ["Appam", "Puttu", "Avial"],
            "French": ["Croissant", "Ratatouille", "Crème Brûlée"],
            "Spanish": ["Paella", "Tapas", "Churros"],
            "Japanese": ["Sushi", "Ramen", "Tempura"],
            "Chinese": ["Dim Sum", "Fried Rice", "Kung Pao Chicken"],
            "Gujarati": ["Dhokla", "Thepla", "Undhiyu"],
            "Marathi": ["Pav Bhaji", "Misal Pav", "Puran Poli"],
            "Punjabi": ["Sarson da Saag", "Makki di Roti", "Lassi"],
            "Bengali": ["Machher Jhol", "Rasgulla", "Mishti Doi"],
        }

        if predicted_label in cuisine_dict:
            dishes = ", ".join(cuisine_dict[predicted_label])
            st.success(
                f"Since the detected accent/language is **{predicted_label}**, "
                f"you might enjoy trying: **{dishes}** 😋"
            )
        else:
            st.info("Cuisine suggestion not available for this accent/language yet.")

        # --- Download Result ---
        result_df = pd.DataFrame({
            "Source": [audio_source],
            "Predicted Accent/Language": [predicted_label],
            "Confidence (%)": [f"{confidence:.2f}"]
        })

        st.download_button(
            label="📄 Download Result (CSV)",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="accent_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
