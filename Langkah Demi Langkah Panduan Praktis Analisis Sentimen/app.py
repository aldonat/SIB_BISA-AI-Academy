import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Memuat model dan vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Fungsi untuk pra-pemrosesan teks (sama seperti yang digunakan saat pelatihan)
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    processed_text = ' '.join(tokens)
    return processed_text

# Antarmuka Streamlit
st.title('Analisis Sentimen dengan Python')
user_input = st.text_area('Masukkan teks:')

if st.button('Analisis'):
    # Pra-pemrosesan input pengguna
    processed_input = preprocess(user_input)
    input_vectorized = vectorizer.transform([processed_input])
    result = model.predict(input_vectorized)
    
    if result[0] == 1:
        st.write('Hasil Sentimen: Positif')
    else:
        st.write('Hasil Sentimen: Negatif')
