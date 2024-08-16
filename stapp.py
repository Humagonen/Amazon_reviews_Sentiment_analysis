
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import requests

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/review_amazon_sentiment5.h5')

model = load_model()

# load tokenizer from google drive
def download_tokenizer():
    url = "https://drive.google.com/file/d/1HDLiJr5UAmpTrpIhtCwofEy09jy0hwRZ/view?usp=drive_link"
    response = requests.get(url)
    with open('tokenizer.json', 'wb') as f:
        f.write(response.content)
    with open('tokenizer.json', 'r') as f:
        tokenizer = json.load(f)
    return tokenizer

tokenizer = download_tokenizer()

# Parameters
num_words = 15000
max_tokens = 166

# Streamlit app
st.title("Sentiment Analysis with LSTM")

# Section 1: Single Text Sentiment Analysis
st.header("Single Text Sentiment Classification")
input_text = st.text_area("Enter the text to classify", "")

if st.button("Classify Sentiment"):
    if input_text:
        # Tokenize and pad the input text
        input_sequence = tokenizer.texts_to_sequences([input_text])
        input_padded = pad_sequences(input_sequence, maxlen=max_tokens)

        # Predict sentiment
        prediction = model.predict(input_padded)[0][0]
        sentiment = "Negative" if prediction > 0.5 else "Positive"

        st.write(f"Predicted Sentiment: **{sentiment}**")
        st.write(f"Probability of being negative: {prediction:.2f}")

# Section 2: Batch Sentiment Analysis
st.header("Batch Sentiment Classification")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the CSV
    df = pd.read_csv(uploaded_file)

    # Ensure the dataframe has the correct columns
    if 'text' in df.columns:
        st.write("Analyzing data...")

        # Tokenize and pad the text data (No need to fit the tokenizer again)
        sequences = tokenizer.texts_to_sequences(df['text'].values)
        padded_sequences = pad_sequences(sequences, maxlen=max_tokens)

        # Predict sentiment
        predictions = model.predict(padded_sequences)
        df['sentiment_score'] = predictions
        df['sentiment'] = df['sentiment_score'].apply(lambda x: "Negative" if x > 0.5 else "Positive")

        # Display the dataframe
        st.write(df.head())

        # Download the resulting dataframe
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download sentiment analysis as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
    else:
        st.error("CSV must have a 'text' column.")
