import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import datetime
import os

from io import BytesIO
with tab2:
    st.write("Click the microphone below and speak a short sentence 🎙️")
    recorded_audio = st_audiorec()
    if recorded_audio is not None:
        audio_file = BytesIO(recorded_audio)
        st.audio(audio_file, format="audio/wav")

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
# Streamlit UI
# -----------------------------
st.title("🎧 Accent / Language Identification App")
st.write("Upload a `.wav` file or record your voice below to detect the accent or language.")

# --- Tabs for Upload or Record ---
tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Voice"])

audio_file = None
uploaded_file = None  # Track upload status

# -----------------------------
# OPTION 1: Upload Audio File
# -----------------------------
with tab1:
    uploaded_file = st.file_uploader("Choose a `.wav` file", type=["wav"])
    if uploaded_file is not None:
        audio_file = uploaded_file
        st.audio(uploaded_file, format="audio/wav")

# -----------------------------
# OPTION 2: Record Audio
# -----------------------------
with tab2:
    st.write("Click the microphone below and speak a short sentence 🎙️")
    recorded_audio = st_audiorec()
    if recorded_audio is not None:
        audio_file = BytesIO(recorded_audio)
        st.audio(audio_file, format="audio/wav")

# -----------------------------
# PROCESS AUDIO IF AVAILABLE
# -----------------------------
if audio_file is not None:
    audio_data, sr = sf.read(audio_file)

    # --- Visualization ---
    #st.subheader("🎶 Audio Visualization")

    # Waveform
    #fig, ax = plt.subplots()
    #librosa.display.waveshow(audio_data, sr=sr, ax=ax, color="#007bff")
    #ax.set_title("Waveform", fontsize=12)
    #st.pyplot(fig)

    # Spectrogram
    #X = librosa.stft(audio_data.astype(float))
    #Xdb = librosa.amplitude_to_db(abs(X))
    #fig, ax = plt.subplots()
    #librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
    #ax.set_title("Spectrogram", fontsize=12)
    #st.pyplot(fig)

    # --- Feature Extraction ---
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    # Ensure audio is long enough
    if len(audio_data) < sr * 0.5:
        st.warning("⚠️ Audio too short! Please record at least 1 second.")
        st.stop()
    # Extract MFCC features (13 coefficients)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)

    # Average each MFCC coefficient across time
    mfcc_mean = np.mean(mfcc, axis=1)

    # Reshape to (1, 13)
    mfcc_scaled = mfcc_mean.reshape(1, -1)
    
    # --- Prediction ---
    try:
        features_scaled = scaler.transform(mfcc_scaled)
        prediction = model.predict(features_scaled)
        predicted_label = encoder.inverse_transform(prediction)[0]

        confidence = (
            np.max(model.predict_proba(features_scaled)) * 100
            if hasattr(model, "predict_proba")
            else 0
        )
        log_data = {
            "timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "source": ["Uploaded" if uploaded_file else "Recorded"],
            "predicted_language": [predicted_label],
            "confidence": [f"{confidence:.2f}"]
        }
        log_df = pd.DataFrame(log_data)
        if not os.path.exists("user_predictions.csv"):
            log_df.to_csv("user_predictions.csv", index=False)
        else:
            log_df.to_csv("user_predictions.csv", mode="a", header=False, index=False)
    
    
        # --- Stylish Result Card ---
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
            "English": ["grilled chciken", "tacos", "pizza", "burger"],
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
            "Source": ["Uploaded" if uploaded_file else "Recorded"],
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

