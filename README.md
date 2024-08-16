# Amazon Reviews Sentiment Analysis and Deployment

deployed streamlit app from huggingface: https://huggingface.co/spaces/hgonen/amazon-reviews-sentiment-analyzer


## **1. Introduction**

Sentiment analysis is a crucial task in natural language processing, involving the classification of text into positive or negative sentiment categories. This report outlines the process of developing a sentiment analysis model using Long Short-Term Memory (LSTM) networks and deploying it as a web application using Streamlit and Hugging Face Spaces.
    
## **2. Data Collection**
    
### **2.1 Source of Data**
The data used in this project was sourced from Kaggle, specifically from a dataset containing Amazon customer reviews. The dataset was provided in a compressed `bz2` format, which was converted to CSV for easier processing and analysis.
    
### **2.2 Data Description**
The dataset consisted of approximately **3.6 million** entries and contained two columns:
    - **`text`**: The review content written by the customer.
    - **`label`**: The sentiment label, where `1` indicates a negative sentiment and `0` indicates a positive sentiment.
    
 The large size of the dataset made it suitable for training a robust deep learning model, particularly well-suited for tasks requiring the understanding of complex patterns in text data.
    
 ## **3. Data Preprocessing**
    
 ### **3.1 Tokenization and Padding**
Given that the LSTM model works well with sequential data, tokenization and padding were key steps in the preprocessing phase:
    - **Tokenization**: Each review text was converted into a sequence of integers, with each integer representing a specific word in the vocabulary.
    - **Padding**: To ensure uniform input size, the tokenized sequences were padded to a maximum length of 166 tokens. This padding step is crucial for LSTM models, which require inputs of the same length.
    
Unlike traditional machine learning models, deep learning models like LSTM can handle raw text data without extensive preprocessing, relying on the embedding and sequential learning capabilities to capture the necessary patterns.
    
## **4. Model Development**
    
 ### **4.1 Model Architecture**
 The LSTM model architecture was designed to effectively handle the sequential nature of text data. The architecture is composed of the following layers:
    
 - **Embedding Layer**: Converts integer-encoded words into dense vectors of size 50, representing the words in a continuous vector space.
      
    - **Dropout Layer**: A dropout rate of 20% was applied to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
      
    - **Bidirectional LSTM Layers**: 
      - The first LSTM layer has 48 units and is bidirectional, meaning it processes the input sequences in both forward and backward directions. This helps the model capture context from both past and future tokens.
      - A second LSTM layer with 24 units, also bidirectional, was added to further refine the learned patterns.
      - A final LSTM layer with 12 units was used to capture the most critical patterns before passing them to the dense layer.
      
    - **Dense Layer**: A fully connected layer with a sigmoid activation function to output the probability of the input text being negative (1) or positive (0).
    
    ### **4.2 Model Compilation**
    The model was compiled with the following settings:
    - **Loss Function**: Binary cross-entropy, as the task involves binary classification.
    - **Optimizer**: Adam optimizer with a learning rate of 0.01 was used for efficient training.
    - **Metrics**: Recall was selected as the primary metric, given its importance in correctly identifying negative sentiments.
    
    ### **4.3 Model Training**
    The model was trained using the following configuration:
    - **Batch Size**: 256, which was chosen to balance training speed and model performance.
    - **Epochs**: The model was trained for 5 epochs.
    - **Early Stopping**: The training process included an early stopping mechanism, which monitored the validation recall metric. If the recall did not improve for 2 consecutive epochs, the training would stop, and the best model weights would be restored.
    
    The model was trained on 2.88 million entries (80% of the dataset) and validated on 720,000 entries (20% of the dataset).
    
    ### **4.4 Model Evaluation**
    The model was evaluated on both the training and test datasets, yielding the following results:
    
    - **Test Data Performance**:
      - **Precision**: 0.93 for class 0, 0.95 for class 1.
      - **Recall**: 0.95 for class 0, 0.93 for class 1.
      - **F1-Score**: 0.94 for both classes.
      - **Accuracy**: 94% on the test data.
    
    - **Train Data Performance**:
      - **Precision**: 0.94 for class 0, 0.95 for class 1.
      - **Recall**: 0.95 for class 0, 0.94 for class 1.
      - **F1-Score**: 0.94 for both classes.
      - **Accuracy**: 94% on the training data.
    
    - **Overall Performance**:
      - The model achieved an average precision-recall score of **0.98**, indicating high reliability in classifying sentiments.
    
    ### **4.5 Model Saving**
    The trained model and tokenizer were saved for future use:
    - **Model**: Saved as `'review_amazon_sentiment5.h5'`.
    - **Tokenizer**: Saved as `'tokenizer.json'`.
    
    ## **5. Model Deployment**
    
    ### **5.1 Streamlit Application**
    To make the sentiment analysis model accessible, a user-friendly web application was built using Streamlit. The app offers two key functionalities:
    - **Single Text Sentiment Analysis**: Allows users to input a single text review and receive an immediate sentiment prediction.
    - **Batch Sentiment Analysis**: Enables users to upload a CSV file containing multiple reviews, with the app classifying the sentiment for each review.
    
    The app also provides visualizations of sentiment distribution (positive vs. negative) using bar charts and pie charts, created with Plotly.
    
    ### **5.2 Deployment on Hugging Face Spaces**
    The Streamlit app was deployed on Hugging Face Spaces, a platform that facilitates the hosting of machine learning models and applications. The deployment process involved:
    1. **Creating a Hugging Face Space**: A new space was set up in Hugging Face.
    2. **Pushing the Code**: The code, including the trained model, tokenizer, and other necessary files, was pushed to the repository.
    3. **Setting Up Requirements**: The `requirements.txt` file was updated to include dependencies such as TensorFlow, Pandas, and Plotly.
    4. **Launching the Application**: The application was launched directly from the Hugging Face platform, making it accessible via a public URL.
    
    ## **6. Conclusion**
    This project demonstrates the end-to-end development and deployment of a sentiment analysis model using LSTM. The deployment of the model via a Streamlit application on Hugging Face Spaces allows for easy access and use by end-users for sentiment analysis tasks. The deep learning approach leveraged in this project underscores the power of LSTM networks in handling text data without the need for extensive preprocessing.
    
    ## **7. Future Work**
    - **Exploring More Advanced Models**: Future work could involve experimenting with transformer-based models like BERT, which might offer improved accuracy.
    - **Expanding Language Support**: Extending the model to handle multiple languages could make it more versatile.
    - **Real-time Data Integration**: Integrating real-time data sources, such as social media feeds, for live sentiment analysis.
