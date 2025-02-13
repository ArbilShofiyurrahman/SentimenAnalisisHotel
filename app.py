import pandas as pd
import re
import joblib
import streamlit as st
from io import BytesIO
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

# Memuat kamus perbaikan
normalized_word = pd.read_excel("kamus perbaikan.xlsx")
normalized_word_dict = {
    str(row['TIDAK BAKU']).strip(): str(row['BAKU']).strip()
    for _, row in normalized_word.iterrows()
    if pd.notna(row['TIDAK BAKU']) and pd.notna(row['BAKU'])
}

# Fungsi normalisasi teks
def normalize_term(document):
    if pd.isna(document):
        return document
    document = str(document)
    for term, replacement in normalized_word_dict.items():
        if term:
            pattern = r'\b' + re.escape(term) + r'\b'
            document = re.sub(pattern, replacement, document, flags=re.IGNORECASE)
    return document

# Fungsi membersihkan teks
def clean_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)  # Menghapus karakter non-alfanumerik
    text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebih
    return text

# Stemming dan stopword remover (PorterStemmer sebagai contoh)
stemmer = PorterStemmer()

def preprocess_text(text):
    # Casefolding
    text = text.lower()
    
    # Cleaning
    text = clean_text(text)
    
    # Normalisasi
    text = normalize_term(text)
    
    # Tokenisasi dan stemming
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Join kembali token
    processed_text = ' '.join(stemmed_tokens)
    return processed_text

# Memuat Model
try:
    tfidf_aspek = joblib.load('tfidf_Aspek.joblib')
    tfidf_fasilitas = joblib.load('tfidfFasilitas.joblib')
    tfidf_pelayanan = joblib.load('tfidfPelayanan.joblib')
    tfidf_masakan = joblib.load('tfidfMasakan.joblib')

    rf_aspek_model = joblib.load('RandomForestAspekModel.joblib')
    rf_fasilitas_model = joblib.load('RandomForestFasilitasModel.joblib')
    rf_pelayanan_model = joblib.load('RandomForestPelayananModel.joblib')
    rf_masakan_model = joblib.load('RandomForestMasakanModel.joblib')
except Exception as e:
    st.error(f"Gagal memuat model atau vektorizer: {e}")
    st.stop()

# Fungsi prediksi sentimen
def predict_sentiment(text, aspect):
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
    
    text_vectorized = vectorizer.transform([text])
    sentiment = model.predict(text_vectorized)[0]
    return sentiment.capitalize()

# Fungsi utama aplikasi
def main():
    st.title("Analisis Sentimen Berbasis Aspek pada Ulasan Hotel")
    st.markdown("Sistem ini memprediksi sentimen berdasarkan aspek ulasan: **Fasilitas**, **Pelayanan**, dan **Masakan**.")

    tab1, tab2 = st.tabs(["Input Teks", "Upload File"])

    with tab1:
        st.subheader("Input Teks Tunggal")
        user_input = st.text_area("Masukkan Teks", placeholder="Contoh: kamar tidak jelek dan rapi")
        if st.button("Prediksi Teks"):
            if not user_input.strip():
                st.warning("Masukkan teks terlebih dahulu.")
            else:
                processed_text = preprocess_text(user_input)
                aspect_vectorized = tfidf_aspek.transform([processed_text])
                predicted_aspect = rf_aspek_model.predict(aspect_vectorized)[0]

                if predicted_aspect == "tidak_dikenali":
                    st.write("*Aspek*: Tidak Dikenali")
                    st.write("*Sentimen*: -")
                else:
                    predicted_sentiment = predict_sentiment(processed_text, predicted_aspect.capitalize())
                    st.write(f"*Aspek*: {predicted_aspect.capitalize()}")
                    st.write(f"*Sentimen*: {predicted_sentiment}")

    with tab2:
        st.subheader("Upload File (CSV atau Excel dengan kolom 'ulasan')")
        uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                
                if 'ulasan' not in df.columns:
                    st.error("File harus memiliki kolom 'ulasan'.")
                else:
                    df = df.dropna(subset=['ulasan']).reset_index(drop=True)
                    df['Ulasan_Preprocessed'] = df['ulasan'].apply(preprocess_text)
                    df['Aspek'] = ""
                    df['Sentimen'] = ""

                    for index, row in df.iterrows():
                        processed_text = row['Ulasan_Preprocessed']
                        aspect_vectorized = tfidf_aspek.transform([processed_text])
                        predicted_aspect = rf_aspek_model.predict(aspect_vectorized)[0]

                        if predicted_aspect == "tidak_dikenali":
                            df.at[index, "Aspek"] = "Tidak Dikenali"
                            df.at[index, "Sentimen"] = "-"
                        else:
                            predicted_sentiment = predict_sentiment(processed_text, predicted_aspect.capitalize())
                            df.at[index, "Aspek"] = predicted_aspect.capitalize()
                            df.at[index, "Sentimen"] = predicted_sentiment
                    
                    st.dataframe(df)
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
                    output.seek(0)
                    
                    st.download_button(
                        "📥 Download Hasil Lengkap (Excel)",
                        data=output,
                        file_name="hasil_analisis_file.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")

if __name__ == "__main__":
    main()
import pandas as pd
import re
import joblib
import streamlit as st
from io import BytesIO
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

# Memuat kamus perbaikan
normalized_word = pd.read_excel("kamus perbaikan.xlsx")
normalized_word_dict = {
    str(row['TIDAK BAKU']).strip(): str(row['BAKU']).strip()
    for _, row in normalized_word.iterrows()
    if pd.notna(row['TIDAK BAKU']) and pd.notna(row['BAKU'])
}

# Fungsi normalisasi teks
def normalize_term(document):
    if pd.isna(document):
        return document
    document = str(document)
    for term, replacement in normalized_word_dict.items():
        if term:
            pattern = r'\b' + re.escape(term) + r'\b'
            document = re.sub(pattern, replacement, document, flags=re.IGNORECASE)
    return document

# Fungsi membersihkan teks
def clean_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)  # Menghapus karakter non-alfanumerik
    text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebih
    return text

# Stemming dan stopword remover (PorterStemmer sebagai contoh)
stemmer = PorterStemmer()

def preprocess_text(text):
    # Casefolding
    text = text.lower()
    
    # Cleaning
    text = clean_text(text)
    
    # Normalisasi
    text = normalize_term(text)
    
    # Tokenisasi dan stemming
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Join kembali token
    processed_text = ' '.join(stemmed_tokens)
    return processed_text

# Memuat Model
try:
    tfidf_aspek = joblib.load('tfidf_Aspek.joblib')
    tfidf_fasilitas = joblib.load('tfidfFasilitas.joblib')
    tfidf_pelayanan = joblib.load('tfidfPelayanan.joblib')
    tfidf_masakan = joblib.load('tfidfMasakan.joblib')

    rf_aspek_model = joblib.load('RandomForestAspekModel.joblib')
    rf_fasilitas_model = joblib.load('RandomForestFasilitas.joblib')
    rf_pelayanan_model = joblib.load('RandomForestPelayananModel.joblib')
    rf_masakan_model = joblib.load('RandomForestMasakanModel.joblib')
except Exception as e:
    st.error(f"Gagal memuat model atau vektorizer: {e}")
    st.stop()

# Fungsi prediksi sentimen
def predict_sentiment(text, aspect):
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
    
    text_vectorized = vectorizer.transform([text])
    sentiment = model.predict(text_vectorized)[0]
    return sentiment.capitalize()

# Fungsi utama aplikasi
def main():
    st.title("Analisis Sentimen Berbasis Aspek pada Ulasan Hotel")
    st.markdown("Sistem ini memprediksi sentimen berdasarkan aspek ulasan: **Fasilitas**, **Pelayanan**, dan **Masakan**.")

    tab1, tab2 = st.tabs(["Input Teks", "Upload File"])

    with tab1:
        st.subheader("Input Teks Tunggal")
        user_input = st.text_area("Masukkan Teks", placeholder="Contoh: kamar tidak jelek dan rapi")
        if st.button("Prediksi Teks"):
            if not user_input.strip():
                st.warning("Masukkan teks terlebih dahulu.")
            else:
                processed_text = preprocess_text(user_input)
                aspect_vectorized = tfidf_aspek.transform([processed_text])
                predicted_aspect = rf_aspek_model.predict(aspect_vectorized)[0]

                if predicted_aspect == "tidak_dikenali":
                    st.write("*Aspek*: Tidak Dikenali")
                    st.write("*Sentimen*: -")
                else:
                    predicted_sentiment = predict_sentiment(processed_text, predicted_aspect.capitalize())
                    st.write(f"*Aspek*: {predicted_aspect.capitalize()}")
                    st.write(f"*Sentimen*: {predicted_sentiment}")

    with tab2:
        st.subheader("Upload File (CSV atau Excel dengan kolom 'ulasan')")
        uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                
                if 'ulasan' not in df.columns:
                    st.error("File harus memiliki kolom 'ulasan'.")
                else:
                    df = df.dropna(subset=['ulasan']).reset_index(drop=True)
                    df['Ulasan_Preprocessed'] = df['ulasan'].apply(preprocess_text)
                    df['Aspek'] = ""
                    df['Sentimen'] = ""

                    for index, row in df.iterrows():
                        processed_text = row['Ulasan_Preprocessed']
                        aspect_vectorized = tfidf_aspek.transform([processed_text])
                        predicted_aspect = rf_aspek_model.predict(aspect_vectorized)[0]

                        if predicted_aspect == "tidak_dikenali":
                            df.at[index, "Aspek"] = "Tidak Dikenali"
                            df.at[index, "Sentimen"] = "-"
                        else:
                            predicted_sentiment = predict_sentiment(processed_text, predicted_aspect.capitalize())
                            df.at[index, "Aspek"] = predicted_aspect.capitalize()
                            df.at[index, "Sentimen"] = predicted_sentiment
                    
                    st.dataframe(df)
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
                    output.seek(0)
                    
                    st.download_button(
                        "📥 Download Hasil Lengkap (Excel)",
                        data=output,
                        file_name="hasil_analisis_file.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")

if __name__ == "__main__":
    main()
