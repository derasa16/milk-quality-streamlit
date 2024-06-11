import pickle
import numpy as np
import streamlit as st
import os

# Title of the web app
st.title("Prediksi Kualitas Susu")

# Input fields for user data
col1, col2 = st.columns(2)
with col1:
    pH = st.text_input("pH")
    if pH != '':
        pH = float(pH)  # Convert to float
with col2:
    Temprature = st.text_input("Temperature")
    if Temprature != '':
        Temprature = float(Temprature)  # Convert to float
with col1:
    Taste = st.text_input("Taste: 1=Baik (enak), 0=Buruk (tidak enak)")
    if Taste != '':
        Taste = float(Taste)  # Convert to float
with col2:
    Odor = st.text_input("Odor Odor: 1=Baik (segar), 0=Buruk (amis,asam)")
    if Odor != '':
        Odor = float(Odor)  # Convert to float
with col1:
    Lemak = st.text_input("Fat: 1=Baik (lemak yang rendah), 0=Buruk (lemak yang tinggi)")
    if Lemak != '':
        Lemak = float(Lemak)  # Convert to float
with col2:
    Turbidity = st.text_input("Turbidity Turbidity: 1=Baik(jernih), 0=Buruk(keruh)")
    if Turbidity != '':
        Turbidity = float(Turbidity)  # Convert to float
with col1:
    Colour = st.text_input("Color: 0 = Alice Blue, 1 = Ghost White, 2 = Honeydew, 3 = Oldlace,  4 = Snow, 5 = White, 6 = Whitesmoke, 7 = Wildsand, 8 = grey")
    if Colour != '':
        Colour = float(Colour)  # Convert to float

# Load saved model and scaler
try:
    milkquality = pickle.load(open('milkquality_model.pkl', 'rb'))
    scaler = pickle.load(open('Scaler.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e}")
except ModuleNotFoundError as e:
    st.error(f"Modul tidak ditemukan: {e}")
except Exception as e:
    st.error(f"Kesalahan lain terjadi: {e}")

# Prediction code
Prediksi_Susu = ''
if st.button("Prediksi Kualitas Susu"):
    if all(v is not None for v in [pH, Temprature, Taste, Odor, Lemak, Turbidity, Colour]):
        try:
            # Scaling numerical input features
            scaled_features = scaler.transform([[pH, Temprature]])

            # Combining all features into a single array
            features = np.array([scaled_features[0][0], scaled_features[0][1], Taste, Odor, Lemak, Turbidity, Colour]).reshape(1, -1)

            # Making prediction with Decision Tree
            Prediksi_Susu = milkquality.predict(features)

            # Interpreting the prediction result
            if Prediksi_Susu[0] == 0:
                Prediksi_Susu = "high"
            elif Prediksi_Susu[0] == 1:
                Prediksi_Susu = "low"
            elif Prediksi_Susu[0] == 2:
                Prediksi_Susu = "medium"
            else:
                Prediksi_Susu = "tidak ditemukan jenis kualitas susu"

            st.success(Prediksi_Susu)
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam prediksi: {e}")
    else:
        st.error("Harap isi semua bidang input sebelum melakukan prediksi")
