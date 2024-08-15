import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Load the model
model = tf.keras.models.load_model('/content/drive/MyDrive/amazon reviews/review_amazon_sentiment.h5')

# Load the tokenizer
with open('/content/drive/MyDrive/amazon reviews/tokenizer.json', 'r') as f:
    data = json.load(f)

data_str = json.dumps(data)
tokenizer = tokenizer_from_json(data_str)

# Parameters
num_words = 15000
max_tokens = 162

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

        # Tokenize and pad the text data
        tokenizer.fit_on_texts(df['text'].values)
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
