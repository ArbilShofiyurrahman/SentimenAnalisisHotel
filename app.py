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

# Fungsi Normalisasi
def normalize_negation(text):
    negation_patterns = {
       r'\btidak bersih\b': 'kotor',
        r'\tidak bagus\b': 'jelek',
        r'\btidak teratur\b': 'berantakan',
        r'\btidak lengkap\b': 'tidaklengkap',
        r'\btidak memadai\b': 'kurangmemadai',
        r'\btidak nyaman\b': 'tidaknyaman',
        r'\btidak ramah\b': 'tidakramah',
        r'\btidak segar\b': 'tidaksegar',
        r'\btidak enak\b': 'tidakenak',
        r'\btidak sopan\b': 'tidaksopan',
        r'\btidak profesional\b': 'tidakprofesional',
        r'\btidak responsif\b': 'cuek',
        r'\btidak efisien\b': 'tidakefisien',
        r'\btidak konsisten\b': 'tidakkonsisten',
        r'\btidak stabil\b': 'tidakstabil',
        r'\btidak matang\b': 'tidakmatang',
        r'\btidak membantu\b': 'tidakmembantu',
        r'\btidak cepat\b': 'lambat',
        r'\btidak wajar\b': 'aneh',
        r'\btidak sesuai\b': 'tidaksesuai',
        r'\btidak aman\b': 'tidakaman',
        r'\btidak jujur\b': 'tidakjujur',
        r'\btidak peduli\b': 'cuek',
        r'\btidak terawat\b': 'tidakterawat',
        r'\btidak tepat waktu\b': 'tidaktepatwaktu',
        r'\btidak tanggap\b': 'tidaksigap',
        r'\btidak bertanggung jawab\b': 'tidakbertanggungjawab',
        r'\btidak wangi\b': 'bau',
        r'\btidak layak\b': 'tidaklayak',
        r'\btidak bisa\b': 'tidakbisa',
        r'\btidak rapi\b': 'tidakrapi',
        r'\btidak jelek\b': 'bagus',

        # Kata negasi diawali dengan "kurang"
        r'\bkurang bersih\b': 'kotor',
        r'\bkurang bagus\b': 'kotor',
        r'\bkurang baik\b': 'kurangbaik',
        r'\bkurang lengkap\b': 'tidaklengkap',
        r'\bkurang memuaskan\b': 'tidakmemuaskan',
        r'\bkurang sopan\b': 'tidaksopan',
        r'\bkurang cepat\b': 'lambat',
        r'\bkurang nyaman\b': 'tidaknyaman',
        r'\bkurang ramah\b': 'tidakramah',
        r'\bkurang segar\b': 'tidaksegar',
        r'\bkurang profesional\b': 'tidakprofesional',
        r'\bkurang terawat\b': 'tidakterawat',
        r'\bkurang efisien\b': 'tidakefisien',
        r'\bkurang matang\b': 'tidakmatang',
        r'\bkurang sigap\b': 'tidaksigap',
        r'\bkurang informatif\b': 'tidakinformatif',
        r'\bkurang sesuai ekspektasi\b': 'kecewa',
        r'\bkurang teratur\b': 'berantakan',
        r'\bkurang konsisten\b': 'tidakkonsisten',
        r'\bkurang memadai\b': 'kurangmemadai',
        r'\bkurang responsif\b': 'cuek',
        r'\bkurang stabil\b': 'tidakstabil',
        r'\bkurang membantu\b': 'tidakmembantu',
        r'\bkurang wajar\b': 'aneh',
        r'\bkurang aman\b': 'tidakaman',
        r'\bkurang peduli\b': 'cuek',
        r'\bkurang tepat waktu\b': 'tidaktepatwaktu',
        r'\bkurang tanggap\b': 'tidaksigap',
        r'\bkurang wangi\b': 'bau',
        r'\bkurang layak\b': 'tidaklayak',
        r'\bkurang enak\b': 'tidakenak',
        r'\bkurang jujur\b': 'tidakjujur',
        r'\bkurang bertanggung jawab\b': 'tidakbertanggungjawab',
        r'\bkurang tepat\b': 'tidaktepat',
        r'\bkurang informatif\b': 'tidakinformatif',
        r'\bkurang terorganisir\b': 'asal',
        r'\bkurang detail\b': 'tidakdetail',
        r'\bkurang memenuhi harapan\b': 'kecewa',

        # Slang dan typo
        r'\bg layak\b': 'tidaklayak',
        r'\bgk layak\b': 'tidaklayak',
        r'\bgak layak\b': 'tidaklayak',
        r'\btdk layak\b': 'tidaklayak',
        r'\bkurg layak\b': 'tidaklayak',
        r'\bkrg layak\b': 'tidaklayak',
        r'\btdk bersih\b': 'kotor',
        r'\bg bersih\b': 'kotor',
        r'\bga bersih\b': 'kotor',
        r'\bgk bersih\b': 'kotor',
        r'\bgak bersih\b': 'kotor',
        r'\btdk bersih\b': 'kotor',
        r'\bkurg bersih\b': 'kotor',
        r'\bkurg bagus\b': 'kotor',
        r'\bkrg bagus\b': 'kotor',
        r'\bg bagus\b': 'kotor',
        r'\bga bagus\b': 'kotor',
        r'\bgak bagus\b': 'kotor',
        r'\bgk bagus\b': 'kotor',
        r'\btdk bagus\b': 'kotor',
        r'\btdk lengkap\b': 'tidaklengkap',
        r'\btdk lgkp\b': 'tidaklengkap',
        r'\btidak lgkp\b': 'tidaklengkap',
        r'\bg lengkap\b': 'tidaklengkap',
        r'\bga lengkap\b': 'tidaklengkap',
        r'\bgak lengkap\b': 'tidaklengkap',
        r'\bgk lengkap\b': 'tidaklengkap',
        r'\bkurg lengkap\b': 'tidaklengkap',
        r'\bg sesuai\b': 'gasesuai',
        r'\bga sesuai\b': 'gasesuai',
        r'\bgak sesuai\b': 'gasesuai',
        r'\bgk sesuai\b': 'gasesuai',
        r'\btdk sesuai\b': 'gasesuai',
        r'\bkurg sesuai\b': 'gasesuai',
        r'\btdk cepat\b': 'lambat',
        r'\bga cepat\b': 'lambat',
        r'\bgak cepat\b': 'lambat',
        r'\bgk cepat\b': 'lambat',
        r'\bkurg cepat\b': 'lambat',
        r'\btdk nyaman\b': 'tidaknyaman',
        r'\bgk nyaman\b': 'tidaknyaman',
        r'\bgak nyaman\b': 'tidaknyaman',
        r'\bg nyaman\b': 'tidaknyaman',
        r'\bgkurg nyaman\b': 'tidaknyaman',
        r'\btdk ramah\b': 'tidakramah',
        r'\bga ramah\b': 'tidakramah',
        r'\bgak ramah\b': 'tidakramah',
        r'\bgk ramah\b': 'tidakramah',
        r'\bkrg ramah\b': 'tidakramah',
        r'\bkurg ramah\b': 'tidakramah',
        r'\bg enak\b': 'tidakenak',
        r'\bga enak\b': 'tidakenak',
        r'\bgk enak\b': 'tidakenak',
        r'\bgak enak\b': 'tidakenak',
        r'\btdk enak\b': 'tidakenak',
        r'\bkurg enak\b': 'tidakenak',
        r'\bkrg enak\b': 'tidakenak',
        r'\bg aman\b': 'tidakaman',
        r'\bga aman\b': 'tidakaman',
        r'\bgak aman\b': 'tidakaman',
        r'\bgk aman\b': 'tidakaman',
        r'\btdk aman\b': 'tidakaman',
        r'\bkurg aman\b': 'tidakaman',
        r'\bkrg aman\b': 'tidakaman',
        r'\bg sopan\b': 'tidaksopan',
        r'\bga sopan\b': 'tidaksopan',
        r'\bgak sopan\b': 'tidaksopan',
        r'\bgk sopan\b': 'tidaksopan',
        r'\btdk sopan\b': 'tidaksopan',
        r'\bkurg sopan\b': 'tidaksopan',
        r'\bkrg sopan\b': 'tidaksopan',
        r'\bg stabil\b': 'tidakstabil',
        r'\bga stabil\b': 'tidakstabil',
        r'\bgak stabil\b': 'tidakstabil',
        r'\bgk stabil\b': 'tidakstabil',
        r'\btdk stabil\b': 'tidakstabil',
        r'\bkurg stabil\b': 'tidakstabil',
        r'\bkrg stabil\b': 'tidakstabil',
        r'\bg rapi\b': 'berantakan',
        r'\bga rapi\b': 'berantakan',
        r'\bgak rapi\b': 'berantakan',
        r'\bgk rapi\b': 'berantakan',
        r'\btdk rapi\b': 'berantakan',
        r'\bkurg rapi\b': 'berantakan',
        r'\bkrg rapi\b': 'berantakan',
        r'\bg profesional\b': 'tidakprofesional',
        r'\bga profesional\b': 'tidakprofesional',
        r'\bgak profesional\b': 'tidakprofesional',
        r'\bgk profesional\b': 'tidakprofesional',
        r'\btdk profesional\b': 'tidakprofesional',
        r'\bkurg profesional\b': 'tidakprofesional',
        r'\bkrg profesional\b': 'tidakprofesional',
        r'\bg efisien\b': 'tidakefisien',
        r'\bga efisien\b': 'tidakefisien',
        r'\bgak efisien\b': 'tidakefisien',
        r'\bgk efisien\b': 'tidakefisien',
        r'\btdk efisien\b': 'tidakefisien',
        r'\bkurg efisien\b': 'tidakefisien',
        r'\bkrg efisien\b': 'tidakefisien',
        r'\btdk konsisten\b': 'tidakkonsisten',
        r'\bgak konsisten\b': 'tidakkonsisten',
        r'\bgk konsisten\b': 'tidakkonsisten',
        r'\bg konsisten\b': 'tidakkonsisten',
        r'\btdk matang\b': 'tidakmatang',
        r'\bg matang\b': 'tidakmatang',
        r'\bgak matang\b': 'tidakmatang',
        r'\bgk matang\b': 'tidakmatang',
        r'\bga matang\b': 'tidakmatang',
        r'\bkurg matang\b': 'tidakmatang',
        r'\bkrg matang\b': 'tidakmatang',
        r'\btdk membantu\b': 'tidakmembantu',
        r'\bg membantu\b': 'tidakmembantu',
        r'\bga membantu\b': 'tidakmembantu',
        r'\bgak membantu\b': 'tidakmembantu',
        r'\bgk membantu\b': 'tidakmembantu',
        r'\bkurg membantu\b': 'tidakmembantu',
        r'\bkrg membantu\b': 'tidakmembantu',
        r'\btdk wajar\b': 'aneh',
        r'\bg wajar\b': 'aneh',
        r'\bga wajar\b': 'aneh',
        r'\bgak wajar\b': 'aneh',
        r'\bgk wajar\b': 'aneh',
        r'\btdk wajar\b': 'aneh',
        r'\bkrg wajar\b': 'aneh',
        r'\bkurg wajar\b': 'aneh',
        r'\btdk peduli\b': 'cuek',
        r'\bg peduli\b': 'cuek',
        r'\bga peduli\b': 'cuek',
        r'\bgak peduli\b': 'cuek',
        r'\bgk peduli\b': 'cuek',
        r'\btdk peduli\b': 'cuek',
        r'\bkurg peduli\b': 'cuek',
        r'\bkrg peduli\b': 'cuek',
        r'\btdk terawat\b': 'tidakterawat',
        r'\bg terawat\b': 'tidakterawat',
        r'\bgak terawat\b': 'tidakterawat',
        r'\bgk terawat\b': 'tidakterawat',
        r'\bkurg terawat\b': 'tidakterawat',
        r'\bkrg terawat\b': 'tidakterawat',
        r'\bkrg jujur\b': 'tidakjujur',
        r'\bga jujur\b': 'tidakjujur',
        r'\bgak jujur\b': 'tidakjujur',
        r'\bgk jujur\b': 'tidakjujur',
        r'\bg jujur\b': 'tidakjujur',
        r'\btdk jujur\b': 'tidakjujur',
        r'\bg ada\b': 'tidakada',
        r'\bga ada\b': 'tidakada',
        r'\bgak ada\b': 'tidakada',
        r'\bgk ada\b': 'tidakada',
        r'\btdk ada\b': 'tidakada',
        r'\bg bisa\b': 'tidakbisa',
        r'\bga bisa\b': 'tidakbisa',
        r'\bgk bisa\b': 'tidakbisa',
        r'\bgabisa\b': 'tidakbisa',
        r'\btdkbisa\b': 'tidakbisa',

        # Frase tambahan sesuai konteks ulasan
        r'\btidak dilayani\b': 'cuek',
        r'\btdk dilayani\b': 'cuek',
        r'\bkurang perhatian\b': 'cuek',
        r'\btdk sesuai\b': 'kecewa',
        r'\btidak memenuhi harapan\b': 'kecewa',
        r'\btdk memenuhi harapan\b': 'kecewa',
        r'\bg memenuhi harapan\b': 'kecewa',
        r'\btidak sesuai ekspektasi\b': 'kecewa',
        r'\btidak sesuai\b': 'kecewa',
        r'\bkurang terorganisir\b': 'asal',
        r'\bkrg terorganisir\b': 'asal',
        r'\bkrg terorganisir\b': 'asal',
        r'\bkurang tanggung jawab\b': 'tidakbertanggungjawab',
         r'\bkrg bertanggung jawab\b': 'tidakbertanggungjawab',
        r'\bkurang detail\b': 'tidakdetail',
        r'\bkrg detail\b': 'tidakdetail',
        r'\bkrg wangi\b': 'bau',
        r'\btdk wangi\b': 'bau',
        r'\bkrg tepat\b': 'tidaktepat',
        r'\bkrg informatif\b': 'tidakinformatif',
        r'\bkrg detail\b': 'tidakdetail'
    }
    for pattern, replacement in negation_patterns.items():
        text = re.sub(pattern, replacement, text)
    return text

# Fungsi Preprocessing
def preprocess_text(text):
    # Casefolding
    text = text.lower()
    
    # Cleaning
    text = clean_text(text)
    
    # Normalisasi
    text = normalize_negation(text)
    
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

def predict_aspect(text):
    """Prediksi aspek dari teks menggunakan model aspek."""
    text_vectorized = tfidf_aspek.transform([text])
    aspect = rf_aspek_model.predict(text_vectorized)[0]
    return aspect

def predict_sentiment(text):
    """Prediksi sentimen berdasarkan aspek yang diprediksi."""
    aspect = predict_aspect(text)  # Step prediksi aspek

    # Pilih TF-IDF dan model berdasarkan aspek yang diprediksi
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
        return "-", "-"

    # Transformasi teks dan prediksi sentimen
    text_vectorized = vectorizer.transform([text])
    sentiment = model.predict(text_vectorized)[0]
    
    return aspect, sentiment.capitalize()

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
    tab1, tab2 = st.tabs(["Input Ulasan", "Upload File"])

    with tab1:
        st.subheader("Input Ulasan Hotel")
        user_input = st.text_area("Masukkan Ulasan", placeholder="kamar tidak jelek dan rapi")
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
        st.subheader("Pastikan Terdapat Kolom (ulasan)")
        uploaded_file = st.file_uploader("Upload File CSV atau Excel", type=["csv", "xlsx"])
        
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
                    st.subheader("Visualisasi Sentimen Pada Setiap Aspek")
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    aspek_list = ["Fasilitas", "Pelayanan", "Masakan"]
                    colors = ["#4DA6FF", "#FF4D4D"]  # Warna biru untuk positif, merah untuk negatif
                    
                    for i, aspek in enumerate(aspek_list):
                        data = df[df['Aspek'] == aspek]['Sentimen'].value_counts()
                        total_data_aspek = len(df[df['Aspek'] == aspek])
                        if not data.empty:
                            wedges, texts, autotexts = axes[i].pie(
                                data, 
                                labels=data.index, 
                                autopct='%1.1f%%', 
                                colors=[colors[0] if label.lower() == "positif" else colors[1] for label in data.index], 
                                startangle=140
                            )
                            # Tambahkan teks di bawah pie chart
                            axes[i].text(0, -1.2, f"Total data: {total_data_aspek}", ha='center', fontsize=10, color='black')
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
    © 2025 Sistem Analisis Sentimen Hotel. by Streamlit.  
    Arbil Shofiyurrahman.  
    Teknologi yang Digunakan: Python, Scikit-learn, TF-IDF, Random Forest.
    """)

if __name__ == "__main__":
    main()
