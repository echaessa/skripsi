import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('saved_assets/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 30

# Fungsi prediksi
def prediksi_sentimen(teks_list, model, tokenizer, maxlen):
    sequences = tokenizer.texts_to_sequences(teks_list)
    
    # Cek apakah semua kata OOV
    all_oov = all(
        len(seq) == 0 or all(token == tokenizer.word_index.get(tokenizer.oov_token) for token in seq)
        for seq in sequences
    )
    
    if all_oov:
        return "unknown"
    
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post')
    probs = model.predict(padded)
    labels = np.argmax(probs, axis=1)
    label_mapping = {0: 'Negatif', 1: 'Positif'}
    return label_mapping[labels[0]]

# UI Streamlit
st.title("Prediksi Sentimen Teks")
st.write("Masukkan teks di bawah untuk mengetahui sentimen (Positif / Negatif):")

# Pilih model
model_choice = st.radio("Pilih Model:", ["Bi-LSTM", "Bi-GRU"])

# Load model berdasarkan pilihan
if model_choice == "Bi-LSTM":
    model = load_model('model_bilstm__best.keras')
else:
    model = load_model('model_bigru__best.keras')

# Input teks
input_teks = st.text_area("Masukkan Teks:")

# Tombol prediksi
if st.button("Prediksi Sentimen"):
    if input_teks.strip():
        hasil = prediksi_sentimen([input_teks], model, tokenizer, MAX_LEN)
        if hasil == "unknown":
            st.info("Model tidak dapat mengenali kata dalam teks tersebut.")
        else:
            st.success(f"Sentimen Prediksi ({model_choice}): {hasil}")
    else:
        st.warning("Teks input tidak boleh kosong!")
