import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu("Skripsi Oktario", ["Klasifikasi Text", 'Hasil Skripsi'], 
        icons=['body-text', 'google-play'], default_index=0)
    selected

if (selected == 'Hasil Skripsi'):
    st.title("Analisis Sentimen Aplikasi Sirekap")

    st.subheader("Dataset Review Sirekap di Google Play Store")
    dataset = pd.read_csv('./review_sirekap_stemmed.csv')
    ds1 = dataset.head()
    ds2 = dataset.tail()
    new_ds = pd.concat([ds1, ds2])
    st.table(new_ds[["content", "score"]])


    st.subheader("Sebaran 1.000 Rating dari Aplikasi Sirekap")
    st.image('./rating.png')


    st.subheader("Hasil Labeling Manual")
    st.image('./sentimen.png')


    st.subheader("Hasil Tokenizing")
    st.table(new_ds[["content","content_token"]])


    st.subheader("Stemming dengan Sastrawi")
    st.table(new_ds[["content_token", "stemmed"]])


    st.subheader("Text String")
    st.table(new_ds[["content_token", "stemmed", "text_string"]])


    data_akurasi = [
        [90,10,98.90,100,94.52],
        [80,20,98.86,100,93.61],
        [70,30,98.72,100,92.33]]

    df_akurasi = pd.DataFrame(data_akurasi, columns=['data_train', 'data_tes', 'random_forest', 'SVM', 'naive_bayes'])
    st.subheader("Hasil Akurasi 3 Klasifikasi dalam 3x Pengujian")
    st.table(df_akurasi)


    st.subheader("Kata yang Sering Muncul")
    st.image('./common.png')


if (selected == 'Klasifikasi Text'):
    # load save model
    model_nb_sentimen = pickle.load(open("model_sentimen.sav", "rb"))

    tfIdf = TfidfVectorizer()
    loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))

    st.title("Prediksi Ulasan Aplikasi Sirekap")
    
    clean_text = st.text_input("Masukkan Text Ulasan")
    
    sentimen_detection = ''

    if st.button('Hasil Deteksi'):
        sentimen_predict = model_nb_sentimen.predict(loaded_vec.fit_transform([clean_text]))

        if (sentimen_predict == 0):
            sentimen_detection = 'Sentimen Netral'
        elif (sentimen_predict == 1):
            sentimen_detection = 'Sentimen Positif'
        else:
            sentimen_detection = 'Sentimen Negatif'
    
    st.success(sentimen_detection)