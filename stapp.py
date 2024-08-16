import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis with LSTM", page_icon=":smiley:", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/review_amazon_sentiment5.h5')

model = load_model()

# Load the tokenizer
with open('models/tokenizer.json', 'r') as f:
    tokenizer_json = f.read()  # Read the file as a string
tokenizer = tokenizer_from_json(tokenizer_json)

# Parameters
num_words = 15000
max_tokens = 166

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.write("Use the navigation below to explore the app.")
option = st.sidebar.selectbox("Choose a section", ["Home", "Sentiment Analysis"])

# Home Page
if option == "Home":
    st.title("📊 Sentiment Analysis with LSTM")
    st.markdown("""
    Welcome to the **Sentiment Analysis** app! This application allows you to analyze the sentiment of text reviews using an LSTM model.
    **Features:**
    - **Single Text Sentiment Classification:** Enter a single review text to get its sentiment.
    - **Batch Sentiment Classification:** Upload a CSV file to analyze multiple reviews at once.
    Use the sidebar to navigate through the different sections of the app.
    """)

# Single Text Sentiment Analysis
elif option == "Sentiment Analysis":
    st.title("Single Text Sentiment Analysis")
    st.markdown("""
    Enter the text below to classify its sentiment. The model will predict whether the sentiment is **Positive** or **Negative**.
    """)
    
    input_text = st.text_area("Enter the text to classify", "")

    if st.button("Classify Sentiment", key='single_text'):
        if input_text:
            # Tokenize and pad the input text
            input_sequence = tokenizer.texts_to_sequences([input_text])
            input_padded = pad_sequences(input_sequence, maxlen=max_tokens)

            # Predict sentiment
            prediction = model.predict(input_padded)[0][0]
            sentiment = "Negative" if prediction > 0.5 else "Positive"

            st.markdown(f"**Predicted Sentiment:** {sentiment}")
            st.markdown(f"**Probability of being negative:** {prediction:.2f}")

    # Batch Sentiment Analysis
    st.title("Batch Sentiment Analysis")
    st.markdown("""
    Upload a CSV file containing review texts to classify their sentiment in bulk.
    Ensure the CSV file has a column named **'text'** which contains the review texts.
    """)
    
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
            st.write("### Analysis Results")
            st.dataframe(df.head())

            # Plot the sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            sentiment_df = pd.DataFrame({
                'Sentiment': sentiment_counts.index,
                'Count': sentiment_counts.values
            })
            
            # Bar Chart
            fig_bar = px.bar(sentiment_df, x='Sentiment', y='Count', title="Sentiment Distribution",
                             labels={'Sentiment': 'Sentiment', 'Count': 'Number of Reviews'},
                             color='Sentiment', color_discrete_map={"Positive": "green", "Negative": "red"})
            
            # Pie Chart
            fig_pie = px.pie(sentiment_df, names='Sentiment', values='Count', title="Sentiment Distribution",
                             color='Sentiment', color_discrete_map={"Positive": "green", "Negative": "red"})
            
            # Display charts side by side
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_bar)
            with col2:
                st.plotly_chart(fig_pie)

            # Download the resulting dataframe
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download sentiment analysis as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
                key='download_button'
            )
        else:
            st.error("CSV must have a 'text' column.")
