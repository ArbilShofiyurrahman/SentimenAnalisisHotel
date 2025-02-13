import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np 
from io import BytesIO
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Initialize Sastrawi components
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Menghapus angka dan karakter non-huruf
    return text

# Load normalization dictionary from Excel
@st.cache_data
def load_normalization_dict():
    try:
        normalized_word = pd.read_excel("kamus perbaikan.xlsx")
        return {str(row['TIDAK BAKU']).strip(): str(row['BAKU']).strip() 
                for _, row in normalized_word.iterrows()}
    except Exception as e:
        st.error(f"Error loading normalization dictionary: {e}")
        return {}

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Menghapus angka dan karakter non-huruf
    return text

# Fungsi Normalisasi
def normalize_text(text):
    if pd.isna(text):
        return text
    
    text = str(text)
    normalized_dict = load_normalization_dict()
    
    for term, replacement in normalized_dict.items():
        pattern = r'\b' + re.escape(term) + r'\b'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

# Fungsi Preprocessing
def preprocess_text(text):
    # Casefolding
    text = text.lower()
    
    # Cleaning
    text = clean_text(text)
    
    # Normalisasi
    text = normalize_text(text)
    
    # Stopword Removal
    text = stopword_remover.remove(text)
    
    # Tokenisasi (split into words)
    tokens = text.split()
    
    # Stemming using Porter Stemmer
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Join back to text
    processed_text = ' '.join(stemmed_tokens)
    
    return processed_text

# Memuat Model
try:
    # Load TF-IDF vectorizers
    tfidf_aspek = joblib.load('tfidf_Aspek.joblib')
    tfidf_fasilitas = joblib.load('tfidfFasilitas.joblib')
    tfidf_pelayanan = joblib.load('tfidfPelayanan.joblib')
    tfidf_masakan = joblib.load('tfidfMasakan.joblib')
    
    # Load Random Forest models
    rf_aspek_model = joblib.load('RandomForestAspekModel.joblib')
    rf_fasilitas_model = joblib.load('RandomForestFasilitas.joblib')
    rf_pelayanan_model = joblib.load('RandomForestPelayananModel.joblib')
    rf_masakan_model = joblib.load('RandomForestMasakanModel.joblib')
except Exception as e:
    st.error(f"Gagal memuat model atau vektorizer: {e}")
    st.stop()

def predict_sentiment(text, aspect):
    # Select appropriate model and vectorizer based on aspect
    if aspect == "Fasilitas":
        vectorizer = tfidf_fasilitas
        model = rf_fasilitas_model
    elif aspect == "Pelayanan":
        vectorizer = tfidf_pelayanan
        model = rf_pelayanan_model
    elif aspect == "Masakan":
        vectorizer = tfidf_masakan
        model = rf_masakan_model
    else:
        return "-"
    
    # Transform text and predict
    text_vectorized = vectorizer.transform([text])
    sentiment = model.predict(text_vectorized)[0]
    return sentiment.capitalize()

def main():
    # Deskripsi Aplikasi
    st.title("Analisis Sentimen Berbasis Aspek pada Ulasan Hotel")
    st.markdown("""
    Sistem Memprediksi Sentimen Berdasarkan Aspek:
    Aplikasi ini menggunakan model machine learning untuk mengklasifikasikan sentimen ulasan menjadi positif atau negatif untuk setiap aspek.
    """)

    # Sidebar untuk Informasi Penting
    st.sidebar.title("Informasi Penting")
    st.sidebar.write("""
    Aspek yang Dianalisis:
    1. Fasilitas : Menganalisis kualitas dan kondisi fasilitas hotel seperti kamar, kolam renang, atau area umum.
    2. Pelayanan : Mengevaluasi kualitas layanan yang diberikan oleh staf hotel, termasuk keramahan dan responsivitas.
    3. Masakan   : Menilai kualitas makanan yang disajikan di restoran hotel atau layanan room service.
    """)
    st.sidebar.write("""
    Sentimen yang Dianalisis:
    1. Positif : Ulasan yang mengandung kata-kata atau frasa yang menunjukkan kepuasan, pujian, atau pengalaman baik.
    2. Negatif : Ulasan yang mengandung kata-kata atau frasa yang menunjukkan ketidakpuasan, kritik, atau pengalaman buruk.
    """)

    # Menyediakan menu/tab untuk input teks atau file
    tab1, tab2 = st.tabs(["Input Teks", "Upload File"])

    with tab1:
        st.subheader("Input Teks Tunggal")
        user_input = st.text_area("Masukkan Teks", placeholder="kamar tidak jelek dan rapi")
        if st.button("Prediksi Teks"):
            if not user_input:
                st.warning("Masukkan teks terlebih dahulu.")
            else:
                # Preprocess text
                processed_text = preprocess_text(user_input)
                
                # Predict aspect
                aspect_vectorized = tfidf_aspek.transform([processed_text])
                predicted_aspect = rf_aspek_model.predict(aspect_vectorized)[0]
                
                if predicted_aspect == "tidak_dikenali":
                    st.write("Aspek: Tidak Dikenali")
                    st.write("Sentimen: -")
                else:
                    # Predict sentiment based on aspect
                    predicted_sentiment = predict_sentiment(processed_text, predicted_aspect.capitalize())
                    st.write(f"Aspek: {predicted_aspect.capitalize()}")
                    st.write(f"Sentimen: {predicted_sentiment}")
    with tab2:
        st.subheader("Input File, Pastikan Terdapat Kolom (ulasan)")
        uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
    
                if 'ulasan' not in df.columns:
                    st.error("File harus memiliki kolom 'ulasan'.")
                else:
                    # Hapus baris dengan ulasan kosong
                    df = df.dropna(subset=['ulasan'])
                    # Hapus baris dengan ulasan yang hanya berisi spasi
                    df = df[df['ulasan'].str.strip() != '']
                    
                    # Reset index setelah menghapus baris
                    df = df.reset_index(drop=True)
                    
                    # Tambahkan kolom untuk ulasan yang telah di preprocessing
                    df["Ulasan_Preprocessed"] = ""
                    df["Aspek"] = ""
                    df["Sentimen"] = ""
                    total_rows = len(df)
    
                    for index, row in df.iterrows():
                        ulasan = str(row['ulasan'])
                        processed_text = preprocess_text(ulasan)
                        
                        # Simpan ulasan yang sudah di preprocessing
                        df.at[index, "Ulasan_Preprocessed"] = processed_text
                        
                        # Predict aspect
                        aspect_vectorized = tfidf_aspek.transform([processed_text])
                        predicted_aspect = rf_aspek_model.predict(aspect_vectorized)[0]
    
                        if predicted_aspect == "tidak_dikenali":
                            df.at[index, "Aspek"] = "Tidak Dikenali"
                            df.at[index, "Sentimen"] = "-"
                        else:
                            # Predict sentiment based on aspect
                            predicted_sentiment = predict_sentiment(processed_text, predicted_aspect.capitalize())
                            df.at[index, "Aspek"] = predicted_aspect.capitalize()
                            df.at[index, "Sentimen"] = predicted_sentiment
    
                    # Tampilkan informasi jumlah data yang diproses
                    st.info(f"Total data setelah menghapus ulasan kosong: {len(df)} baris")
    
                    # Visualisasi Pie Chart
                    st.subheader("Visualisasi Sentimen per Aspek")
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    aspek_list = ["Fasilitas", "Pelayanan", "Masakan"]
                    colors = ["#66b3ff", "#ff9999"]
    
                    for i, aspek in enumerate(aspek_list):
                        data = df[df['Aspek'] == aspek]['Sentimen'].value_counts()
                        if not data.empty:
                            axes[i].pie(data, labels=data.index, autopct='%1.1f%%', colors=colors, startangle=140)
                            axes[i].set_title(f"Aspek {aspek}")
                        else:
                            axes[i].pie([1], labels=["Tidak Ada Data"], colors=["#d3d3d3"])
                            axes[i].set_title(f"Aspek {aspek}")
    
                    st.pyplot(fig)
    
                    # Menampilkan DataFrame hasil prediksi
                    st.subheader("Hasil Analisis")
                    st.dataframe(df)
    
                    # Download hasil
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
                    output.seek(0)
    
                    st.download_button(
                        label="📥 Download Hasil Lengkap (Excel)",
                        data=output,
                        file_name="hasil_analisis_file.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")




    # Footer
    st.markdown("---")
    st.caption("""
    © 2025 Sistem Analisis Sentimen Hotel. Dibangun dengan Streamlit.  
    Dikembangkan oleh Arbil Shofiyurrahman.  
    Teknologi yang Digunakan: Python, Scikit-learn, TF-IDF, Random Forest.
    """)

if _name_ == "_main_":
    main()
