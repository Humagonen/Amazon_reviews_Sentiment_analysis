{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CaQ-DGjVUsmm"
      },
      "source": [
        "# Streamlit Sentiment analysis web app"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fp78bMB74b1_",
        "outputId": "8a7e4d48-0c8d-40ff-ea31-c779265de17d"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ATesuDKB4J62",
        "outputId": "d445ec24-8b45-4a8b-b160-4366c691801b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/8.7 MB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[91m━━━━━━━━━━━━━\u001b[0m\u001b[90m╺\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.8/8.7 MB\u001b[0m \u001b[31m85.5 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[90m╺\u001b[0m\u001b[90m━━━━━━━━━\u001b[0m \u001b[32m6.6/8.7 MB\u001b[0m \u001b[31m94.6 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m8.6/8.7 MB\u001b[0m \u001b[31m95.8 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.7/8.7 MB\u001b[0m \u001b[31m62.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.3/207.3 kB\u001b[0m \u001b[31m16.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m70.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m82.9/82.9 kB\u001b[0m \u001b[31m6.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m5.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[K\u001b[?25h\n",
            "added 22 packages, and audited 23 packages in 2s\n",
            "\n",
            "3 packages are looking for funding\n",
            "  run `npm fund` for details\n",
            "\n",
            "1 \u001b[33m\u001b[1mmoderate\u001b[22m\u001b[39m severity vulnerability\n",
            "\n",
            "To address all issues (including breaking changes), run:\n",
            "  npm audit fix --force\n",
            "\n",
            "Run `npm audit` for details.\n"
          ]
        }
      ],
      "source": [
        "!pip install -q streamlit tensorflow pandas numpy\n",
        "!npm install -g localtunnel"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "E5knd9OO4LIf",
        "outputId": "39ecad71-768c-47e1-c6b3-cb0b0caf7317"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Writing app.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app.py\n",
        "\n",
        "import streamlit as st\n",
        "import tensorflow as tf\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "from tensorflow.keras.preprocessing.text import Tokenizer\n",
        "from tensorflow.keras.preprocessing.text import tokenizer_from_json\n",
        "import json\n",
        "\n",
        "model = tf.keras.models.load_model('/content/drive/MyDrive/amazon reviews/review_amazon_sentiment.h5')\n",
        "\n",
        "\n",
        "# Load the tokenizer JSON content as a string\n",
        "with open('/content/drive/MyDrive/amazon reviews/tokenizer.json', 'r') as f:\n",
        "    data = json.load(f)  # This gives you a dictionary\n",
        "\n",
        "# Convert the dictionary back to a JSON string\n",
        "data_str = json.dumps(data)\n",
        "\n",
        "# Use the string to load the tokenizer\n",
        "tokenizer = tokenizer_from_json(data_str)\n",
        "\n",
        "# model = tf.keras.models.load_model('review_amazon_sentiment.h5')\n",
        "# Parameters\n",
        "num_words = 15000\n",
        "max_tokens = 162\n",
        "\n",
        "\n",
        "# Streamlit app\n",
        "st.title(\"Sentiment Analysis with LSTM\")\n",
        "\n",
        "# Section 1: Single Text Sentiment Analysis\n",
        "st.header(\"Single Text Sentiment Classification\")\n",
        "\n",
        "input_text = st.text_area(\"Enter the text to classify\", \"\")\n",
        "\n",
        "if st.button(\"Classify Sentiment\"):\n",
        "    if input_text:\n",
        "        # Tokenize and pad the input text\n",
        "        input_sequence = tokenizer.texts_to_sequences([input_text])\n",
        "        input_padded = pad_sequences(input_sequence, maxlen=max_tokens)\n",
        "\n",
        "        # Predict sentiment\n",
        "        prediction = model.predict(input_padded)[0][0]\n",
        "        sentiment = \"Negative\" if prediction > 0.5 else \"Positive\"\n",
        "\n",
        "        st.write(f\"Predicted Sentiment: **{sentiment}**\")\n",
        "        st.write(f\"Probability of being negative: {prediction:.2f}\")\n",
        "\n",
        "# Section 2: Batch Sentiment Analysis\n",
        "st.header(\"Batch Sentiment Classification\")\n",
        "uploaded_file = st.file_uploader(\"Upload a CSV file\", type=[\"csv\"])\n",
        "\n",
        "if uploaded_file is not None:\n",
        "    # Load the CSV\n",
        "    df = pd.read_csv(uploaded_file)\n",
        "\n",
        "    # Ensure the dataframe has the correct columns\n",
        "    if 'text' in df.columns:\n",
        "        st.write(\"Analyzing data...\")\n",
        "\n",
        "        # Tokenize and pad the text data\n",
        "        tokenizer.fit_on_texts(df['text'].values)\n",
        "        sequences = tokenizer.texts_to_sequences(df['text'].values)\n",
        "        padded_sequences = pad_sequences(sequences, maxlen=max_tokens)\n",
        "\n",
        "        # Predict sentiment\n",
        "        predictions = model.predict(padded_sequences)\n",
        "        df['sentiment_score'] = predictions\n",
        "        df['sentiment'] = df['sentiment_score'].apply(lambda x: \"Negative\" if x > 0.5 else \"Positive\")\n",
        "\n",
        "        # Display the dataframe\n",
        "        st.write(df.head())\n",
        "\n",
        "        # Download the resulting dataframe\n",
        "        csv = df.to_csv(index=False).encode('utf-8')\n",
        "        st.download_button(\n",
        "            label=\"Download sentiment analysis as CSV\",\n",
        "            data=csv,\n",
        "            file_name='sentiment.csv',\n",
        "            mime='text/csv',\n",
        "        )\n",
        "    else:\n",
        "        st.error(\"CSV must have a 'text' column.\")\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pVWWEiuE4LC9",
        "outputId": "70956fae-0051-4a26-ebc8-64839535cb41"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "34.91.162.48\n",
            "your url is: https://quick-actors-hope.loca.lt\n"
          ]
        }
      ],
      "source": [
        "!streamlit run app.py &>/content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com  # ip is your password"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xuaFWixPUcPY"
      },
      "source": [
        "# Trying out model in notebook before streamlit"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d_E3S4zJ-M6B"
      },
      "outputs": [],
      "source": [
        "from tensorflow.keras.preprocessing.text import tokenizer_from_json\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "import tensorflow as tf\n",
        "import json\n",
        "\n",
        "# Load the tokenizer JSON content as a string\n",
        "with open('/content/drive/MyDrive/amazon reviews/tokenizer.json', 'r') as f:\n",
        "    data = json.load(f)  # This gives you a dictionary\n",
        "\n",
        "# Convert the dictionary back to a JSON string\n",
        "data_str = json.dumps(data)\n",
        "\n",
        "# Use the string to load the tokenizer\n",
        "tokenizer = tokenizer_from_json(data_str)\n",
        "\n",
        "\n",
        "# load model\n",
        "model = tf.keras.models.load_model('/content/drive/MyDrive/amazon reviews/review_amazon_sentiment.h5')\n",
        "\n",
        "\n",
        "# new data\n",
        "review1 = \"I hated this product, never buying it again!\"\n",
        "review2 = \"beautiful! fast shipping and a responsive seller\"\n",
        "review3 = \"garbage product, no one should sell such thing\"\n",
        "review4 = \"great price for a product like this, definitely buying it again\"\n",
        "\n",
        "reviews = [review1, review2, review3, review4]\n",
        "\n",
        "\n",
        "# apply tokenization\n",
        "num_words = 15000\n",
        "max_tokens = 162\n",
        "\n",
        "tokens = tokenizer.texts_to_sequences(reviews)\n",
        "tokens_pad = pad_sequences(tokens, maxlen=max_tokens)\n",
        "tokens_pad.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lpQRdg1y_NwR",
        "outputId": "c8070224-000d-4186-fce9-ba43f5da4234"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 983ms/step\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "array([[1],\n",
              "       [0],\n",
              "       [1],\n",
              "       [0]])"
            ]
          },
          "execution_count": 10,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# prediction\n",
        "\n",
        " (model.predict(tokens_pad) >0.5).astype(\"int\")"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1iKkhOZMLjZKBJDhwXkdvI2RR-iPKFpTW",
      "authorship_tag": "ABX9TyNfSAwvYczwOR4TrP3e6qEu"
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}