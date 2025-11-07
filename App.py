import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Konfigurasi Halaman
# =============================================================================
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📡",
    layout="wide"
)

# =============================================================================
# Memuat Model dan Scaler
# =============================================================================
# @st.cache_resource akan menyimpan file di memori agar tidak di-load ulang
@st.cache_resource
def load_files():
    """
    Memuat model dan scaler.
    """
    try:
        model = joblib.load("cart_model_churn.joblib")
        scaler = joblib.load("scaler_churn.joblib")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: File model/scaler tidak ditemukan. Pastikan 'cart_model_churn.joblib' dan 'scaler_churn.joblib' ada di folder yang sama.")
        st.error(f"Detail: {e}")
        return None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {e}")
        # Menampilkan detail fitur dari scaler yang salah untuk debugging
        if "scaler" in str(e) or "scaler" in locals():
            try:
                st.warning(f"Info fitur dari 'scaler_churn.joblib': {scaler.feature_names_in_}")
            except:
                pass
        return None, None

model, scaler = load_files()

# Jika file tidak ada, hentikan aplikasi
if model is None or scaler is None:
    st.stop()

# =============================================================================
# Peringatan Penting (HAPUS JIKA SUDAH DIPERBAIKI)
# =============================================================================
try:
    # Cek apakah scaler-nya benar
    if 'tenure' not in scaler.feature_names_in_:
        st.warning("""
        **PERINGATAN:** File `scaler_churn.joblib` yang dimuat tampaknya salah! 
        Fiturnya adalah: `{}`, padahal seharusnya `['tenure', 'MonthlyCharges', 'TotalCharges']`.
        
        **Harap ganti file `scaler_churn.joblib` dengan file scaler yang benar dari notebook Praktek Anda agar aplikasi berfungsi dengan benar.**
        """.format(scaler.feature_names_in_))
except:
    pass

# =============================================================================
# UI - Input dari Pengguna
# =============================================================================
st.title("Prediksi Churn Pelanggan 📡")
st.markdown("Masukkan data pelanggan di bawah ini untuk memprediksi potensi churn.")

# Buat layout kolom
col1, col2, col3 = st.columns(3)

# === Kolom 1: Demografi & Penggunaan ===
with col1:
    st.header("Data Pelanggan")
    
    gender = st.selectbox("Gender", ("Male", "Female"))
    
    # SeniorCitizen PENTING - ini ada di model Anda tapi tidak ada di HTML Anda
    SeniorCitizen = st.selectbox("Senior Citizen", ("No", "Yes")) 
    
    Partner = st.selectbox("Partner", ("No", "Yes"))
    Dependents = st.selectbox("Dependen", ("No", "Yes"))

    st.divider()
    
    st.subheader("Data Penggunaan")
    tenure = st.number_input("Tenure (Bulan)", min_value=0, max_value=120, value=12)
    MonthlyCharges = st.number_input("Tagihan Bulanan ($)", min_value=0.0, value=70.50, format="%.2f")
    TotalCharges = st.number_input("Total Tagihan ($)", min_value=0.0, value=1200.00, format="%.2f")

# === Kolom 2: Layanan Telepon & Internet ===
with col2:
    st.header("Layanan")

    PhoneService = st.selectbox("Layanan Telepon", ("Yes", "No"))
    MultipleLines = st.selectbox("Multiple Lines", ("No", "Yes", "No phone service"))
    
    st.divider()

    InternetService = st.selectbox("Layanan Internet", ("DSL", "Fiber optic", "No"))
    OnlineSecurity = st.selectbox("Keamanan Online", ("No", "Yes", "No internet service"))
    OnlineBackup = st.selectbox("Backup Online", ("No", "Yes", "No internet service"))
    DeviceProtection = st.selectbox("Proteksi Perangkat", ("No", "Yes", "No internet service"))
    TechSupport = st.selectbox("Dukungan Teknis", ("No", "Yes", "No internet service"))
    StreamingTV = st.selectbox("Streaming TV", ("No", "Yes", "No internet service"))
    StreamingMovies = st.selectbox("Streaming Film", ("No", "Yes", "No internet service"))

# === Kolom 3: Kontrak & Pembayaran ===
with col3:
    st.header("Kontrak & Pembayaran")
    
    Contract = st.selectbox("Tipe Kontrak", ("Month-to-month", "One year", "Two year"))
    PaperlessBilling = st.selectbox("Tagihan Paperless", ("Yes", "No"))
    PaymentMethod = st.selectbox("Metode Pembayaran", (
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ))
    
    st.divider()
    
    # Tombol Prediksi
    predict_button = st.button("Prediksi Churn", type="primary", use_container_width=True)


# =============================================================================
# Logika Preprocessing dan Prediksi
# =============================================================================
if predict_button:
    
    # 1. Kumpulkan data mentah ke dalam dictionary
    raw_data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod
    }

    # 2. Buat DataFrame dari data mentah
    input_df = pd.DataFrame([raw_data])
    
    # 3. Scaling Fitur Numerik
    # Ini MENGASUMSIKAN 'scaler_churn.joblib' adalah scaler yang BENAR
    # yang dilatih pada ['tenure', 'MonthlyCharges', 'TotalCharges']
    try:
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    except Exception as e:
        st.error(f"Error saat scaling: {e}")
        st.error("Ini kemungkinan besar terjadi karena file 'scaler_churn.joblib' yang salah. Harap ganti dengan file yang benar.")
        st.stop()

    # 4. Encoding Fitur Kategori (Manual One-Hot Encoding)
    # Ini untuk memastikan urutan kolom TEPAT SAMA dengan yang dilihat model
    
    # Dapatkan daftar fitur yang diharapkan model
    model_features = model.feature_names_in_
    
    # Buat DataFrame kosong dengan kolom yang benar
    processed_df = pd.DataFrame(columns=model_features)
    
    # Isi data numerik & biner sederhana
    processed_df['tenure'] = input_df['tenure']
    processed_df['MonthlyCharges'] = input_df['MonthlyCharges']
    processed_df['TotalCharges'] = input_df['TotalCharges']
    processed_df['SeniorCitizen'] = 1 if SeniorCitizen == 'Yes' else 0
    processed_df['gender_Male'] = 1 if gender == 'Male' else 0
    processed_df['Partner_Yes'] = 1 if Partner == 'Yes' else 0
    processed_df['Dependents_Yes'] = 1 if Dependents == 'Yes' else 0
    processed_df['PhoneService_Yes'] = 1 if PhoneService == 'Yes' else 0
    processed_df['PaperlessBilling_Yes'] = 1 if PaperlessBilling == 'Yes' else 0

    # Encoding Multi-Kategori (hati-hati)
    # MultipleLines
    processed_df['MultipleLines_No phone service'] = 1 if MultipleLines == 'No phone service' else 0
    processed_df['MultipleLines_Yes'] = 1 if MultipleLines == 'Yes' else 0
    
    # InternetService
    processed_df['InternetService_Fiber optic'] = 1 if InternetService == 'Fiber optic' else 0
    processed_df['InternetService_No'] = 1 if InternetService == 'No' else 0
    
    # OnlineSecurity
    processed_df['OnlineSecurity_No internet service'] = 1 if OnlineSecurity == 'No internet service' else 0
    processed_df['OnlineSecurity_Yes'] = 1 if OnlineSecurity == 'Yes' else 0
    
    # OnlineBackup
    processed_df['OnlineBackup_No internet service'] = 1 if OnlineBackup == 'No internet service' else 0
    processed_df['OnlineBackup_Yes'] = 1 if OnlineBackup == 'Yes' else 0
    
    # DeviceProtection
    processed_df['DeviceProtection_No internet service'] = 1 if DeviceProtection == 'No internet service' else 0
    processed_df['DeviceProtection_Yes'] = 1 if DeviceProtection == 'Yes' else 0
    
    # TechSupport
    processed_df['TechSupport_No internet service'] = 1 if TechSupport == 'No internet service' else 0
    processed_df['TechSupport_Yes'] = 1 if TechSupport == 'Yes' else 0
    
    # StreamingTV
    processed_df['StreamingTV_No internet service'] = 1 if StreamingTV == 'No internet service' else 0
    processed_df['StreamingTV_Yes'] = 1 if StreamingTV == 'Yes' else 0
    
    # StreamingMovies
    processed_df['StreamingMovies_No internet service'] = 1 if StreamingMovies == 'No internet service' else 0
    processed_df['StreamingMovies_Yes'] = 1 if StreamingMovies == 'Yes' else 0
    
    # Contract
    processed_df['Contract_One year'] = 1 if Contract == 'One year' else 0
    processed_df['Contract_Two year'] = 1 if Contract == 'Two year' else 0
    
    # PaymentMethod
    processed_df['PaymentMethod_Credit card (automatic)'] = 1 if PaymentMethod == 'Credit card (automatic)' else 0
    processed_df['PaymentMethod_Electronic check'] = 1 if PaymentMethod == 'Electronic check' else 0
    processed_df['PaymentMethod_Mailed check'] = 1 if PaymentMethod == 'Mailed check' else 0
    
    # Isi semua kolom yang tidak terisi (misal 'gender_Female') dengan 0
    processed_df = processed_df.fillna(0)
    
    # Pastikan urutan kolom 100% sama
    processed_df = processed_df[model_features]

    # 5. Lakukan Prediksi
    with st.spinner("Menganalisis data..."):
        try:
            prediction = model.predict(processed_df)
            probability = model.predict_proba(processed_df)
            
            # Ambil probabilitas untuk kelas yang diprediksi
            pred_class = prediction[0]
            pred_proba = probability[0][pred_class]

            # 6. Tampilkan Hasil
            st.divider()
            st.header("Hasil Prediksi")
            if pred_class == 1: # Asumsi 1 adalah 'Churn'
                st.error(f"**Prediksi: CHURN**", icon="🚨")
                st.markdown(f"Pelanggan ini memiliki probabilitas **{pred_proba:.1%}** untuk **berhenti berlangganan**.")
                st.markdown("**Rekomendasi:** Segera lakukan intervensi, tawarkan promo, atau hubungi pelanggan untuk menanyakan masalah.")
            else: # Asumsi 0 adalah 'Loyal'
                st.success(f"**Prediksi: LOYAL**", icon="✅")
                st.markdown(f"Pelanggan ini memiliki probabilitas **{pred_proba:.1%}** untuk **tetap loyal**.")
                st.markdown("**Rekomendasi:** Pertahankan kualitas layanan atau tawarkan *upsell* produk lain.")

        except Exception as e:
            st.error(f"Error saat prediksi: {e}")
            st.dataframe(processed_df) # Tampilkan dataframe untuk debug
            st.warning("Pastikan semua nilai input sudah benar.")
