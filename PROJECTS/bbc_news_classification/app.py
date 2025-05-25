import gradio as gr # type: ignore
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle

# Load the saved model and tokenizer
model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    text_tokenizer = pickle.load(handle)

# Define categories (based on the notebook's dataset)
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
max_sentence_len = 250  # Adjust based on notebook's preprocessing

# Function to tokenize and pad input text
def tokenize_and_pad(texts, max_length, tokenizer):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded

# Prediction function
def predict_category(title):
    if not title.strip():
        return "Please enter a valid title."
    sequence = tokenize_and_pad([title], max_length=max_sentence_len, tokenizer=text_tokenizer)
    prediction = model.predict(sequence)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    return f"Predicted Category: {categories[predicted_class_index]}"

# Gradio interface
iface = gr.Interface(
    fn=predict_category,
    inputs=gr.Textbox(lines=2, placeholder="Enter an article title..."),
    outputs="text",
    title="BBC News Article Category Predictor",
    description="Enter a news article title to predict its category (business, entertainment, politics, sport, tech)."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()