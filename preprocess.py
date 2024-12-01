import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence



# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


def preprocess_text(text):
    print(text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    print(cleaned_text)
    words = cleaned_text.lower().split()
    print(words)
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    print(encoded_review)
    encoded_review = [i for i in encoded_review if i<10000]
    print(encoded_review)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review, encoded_review