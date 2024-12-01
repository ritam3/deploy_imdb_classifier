# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import re

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb_2.h5')

# Function to preprocess user input
def preprocess_text(text):
    print(text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    print(cleaned_text)
    words = cleaned_text.lower().split()
    print(words)
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    print(encoded_review)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review, encoded_review


import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('IMDB Movie Review using Simple RNN')

if st.button('Classify'):

    preprocessed_input, encoded_inp =preprocess_text(user_input)
    print(encoded_inp)
    if len(encoded_inp)==0:
        st.write('Could not classify. Try with a different review')
    ## MAke prediction
    else:
        try:
            prediction=model.predict(preprocessed_input)
            sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
            st.write(f'Sentiment: {sentiment}')
            st.write(f'Prediction Score: {prediction[0][0]}')
        except:
            st.write('Could not classify. Try with a different review')
    

    # Display the result
    
else:
    st.write('Please enter a movie review.')

