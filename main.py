# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import preprocess_text

# Load the pre-trained model with ReLU activation
model = load_model('models/simple_rnn_imdb_4.h5')


import streamlit as st


st.title('IMDB Movie Review Sentiment Analysis')
st.header("Model Description")
st.markdown("""
This app demonstrates a Recurrent Neural Network (RNN) model trained for sentiment analysis using IMDB Dataset. The vocabulary was set at 10000 words.
The model architecture is shown below.
""")
st.image("image/model_architecture.png", caption="Model Architecture Overview", use_container_width=True)
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
            print(prediction[0][0])
            sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
            st.write(f'Sentiment: {sentiment}')
            st.write(f'Prediction Score: {prediction[0][0]}')
        except:
            st.write('Could not classify. Try with a different review')
    

    # Display the result
    
else:
    st.write('Please enter a movie review.')

