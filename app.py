import streamlit as st
import pandas as pd
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from nltk.corpus import stopwords

# Load the lemmatizer
with open('lemmatizer.pkl', 'rb') as f:
    lemmatizer = pickle.load(f)

# Load the stop words
with open('stop_words.pkl', 'rb') as f:
    stop_words = pickle.load(f)

# Load the LSTM model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('lstm_model.h5')

# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Initialize the model and tokenizer
model = load_model()
tokenizer = load_tokenizer()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Prediction function
def predict_text(text, model, tokenizer):
    max_len = 100
    preprocessed_text = preprocess_text(text)
    
    # Convert text to sequences using the tokenizer
    input_seq = tokenizer.texts_to_sequences([preprocessed_text])
    
    # Handle case where no valid tokens are found
    if not input_seq or not input_seq[0]:
        st.error("No valid tokens found in the input text. Please ensure the input text contains recognizable words.")
        return None, None

    # Pad the sequence
    input_pad = pad_sequences(input_seq, maxlen=max_len)
    pred_prob = model.predict(input_pad)[0][0]
    
    # Set threshold for classification
    prediction = 1 if pred_prob > 0.1 else 0  
    return prediction, pred_prob

# Streamlit UI
st.title("Sentiment Analysis of Product Feedback")

# Text input for feedback
user_input = st.text_area("Enter your feedback:")

if st.button("Predict"):
    if user_input:
        if model and tokenizer:
            prediction, pred_prob = predict_text(user_input, model, tokenizer)
            
            st.write("**Input text:**", user_input)
            
            if prediction is not None:
                st.write(f"**Prediction:** {'Recommended' if prediction == 1 else 'Not recommended'}")
              
            else:
                st.write("An error occurred during prediction.")
        else:
            st.write("Model or tokenizer could not be loaded. Please check the logs.")
    else:
        st.write("Please enter some feedback to analyze.")

st.write("Developed with ❤️ using Streamlit and TensorFlow.")









