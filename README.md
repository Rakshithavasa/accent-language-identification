# 🎧 Accent / Language Identification App

This project is a **Machine Learning & Audio Processing Web Application** that identifies the **accent or native language** of a speaker from a short voice sample.  
It also provides a **fun cuisine recommendation** based on the detected accent!

---

## 🌍 Project Overview

This app analyzes an audio file (or recorded voice) using **MFCC features** and a trained **Random Forest classifier** to detect the spoken accent/language.  
It’s built using **Streamlit** for the UI and supports both file upload and real-time microphone recording.

---

## 🧠 Features

✅ Upload or record `.wav` audio files  
✅ Predict the accent/language using an ML model  
✅ Shows confidence percentage  
✅ Recommends cuisines based on the detected accent  
✅ Beautiful gradient UI with Material-style cards  

---

## 🧩 Technologies Used

| Component | Description |
|------------|-------------|
| **Frontend** | Streamlit (Python-based Web UI) |
| **Audio Processing** | Librosa, SoundFile |
| **Machine Learning** | Scikit-learn (Random Forest Classifier) |
| **Data Handling** | Pandas, NumPy |
| **Deployment** | Streamlit Cloud |

---

## 🗂️ Dataset

We used the **IndicAccentDB** dataset — a multilingual Indian English speech database.  
Each audio file corresponds to one accent/language label such as:

| Label | Example Language |
|--------|------------------|
| English | Neutral |
| Hindi | North Indian Accent |
| Tamil | South Indian Accent |
| Telugu | South Indian Accent |
| Kannada | South Indian Accent |
| Malayalam | South Indian Accent |
| Bengali | East Indian Accent |
| Gujarati | West Indian Accent |
| Marathi | Central Indian Accent |
| Punjabi | North Indian Accent |

---

## 🧾 Files in This Repository

