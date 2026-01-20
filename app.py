import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(page_title="Heart Disease Detector", page_icon="❤️")

# 1. Judul dan Deskripsi
st.title("Sistem Deteksi Risiko Penyakit Jantung 🫀")
st.markdown("""
Aplikasi ini untuk memprediksi risiko penyakit jantung 
berdasarkan data medis pasien. 
---
""")

# 2. Fungsi Load Model
@st.cache_resource
def load_my_model():
    # Pastikan file 'model_jantung_terbaik.h5' ada di folder yang sama
    return tf.keras.models.load_model('model_jantung_terbaik.h5')

try:
    model = load_my_model()
except:
    st.error("Model '.h5' tidak ditemukan. Pastikan sudah mengupload file model ke GitHub.")

# 3. Input Data Pasien (Sesuai kolom heart.csv)
st.sidebar.header("Input Data Medis")

def user_input_features():
    age = st.sidebar.number_input("Usia (age)", 1, 100, 45)
    sex = st.sidebar.selectbox("Jenis Kelamin (sex)", [1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
    cp = st.sidebar.slider("Tipe Nyeri Dada (cp)", 0, 3, 1)
    trestbps = st.sidebar.number_input("Tekanan Darah (trestbps)", 80, 200, 120)
    chol = st.sidebar.number_input("Kolesterol (chol)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Gula Darah > 120 mg/dl (fbs)", [0, 1])
    restecg = st.sidebar.slider("Hasil EKG Istirahat (restecg)", 0, 2, 0)
    thalach = st.sidebar.number_input("Detak Jantung Maks (thalach)", 60, 220, 150)
    exang = st.sidebar.selectbox("Angina Akibat Olahraga (exang)", [0, 1])
    oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.sidebar.slider("Slope Segmen ST (slope)", 0, 2, 1)
    ca = st.sidebar.slider("Jumlah Pembuluh Darah Utama (ca)", 0, 4, 0)
    thal = st.sidebar.slider("Thalassemia (thal)", 0, 3, 2)
    
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return np.array([list(data.values())])

# Ambil input
input_data = user_input_features()

# 4. Tombol Prediksi
if st.button("Analisis Risiko"):
    # Preprocessing Input: Reshape untuk 1D-CNN (Samples, Features, 1)
    # Catatan: Idealnya gunakan Scaler yang sama dari Colab, namun untuk demo 
    # kita bisa gunakan simple scaling jika tidak membawa objek scaler.bin
    input_reshaped = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
    
    # Prediksi
    prediction = model.predict(input_reshaped)
    probability = prediction[0][0]
    
    # Tampilan Hasil
    st.subheader("Hasil Analisis:")
    if probability > 0.5:
        st.error(f"⚠️ **TERDETEKSI RISIKO TINGGI** (Probabilitas: {probability*100:.2f}%)")
        st.write("Segera konsultasikan hasil ini dengan tenaga medis profesional.")
    else:
        st.success(f"✅ **RISIKO RENDAH** (Probabilitas: {probability*100:.2f}%)")
        st.write("Tetap jaga pola makan dan rutin berolahraga.")

st.info("Catatan: Aplikasi ini hanya untuk tujuan edukasi (Tugas UAS).")
