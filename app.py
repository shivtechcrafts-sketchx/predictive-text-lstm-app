import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from utils import predict_next_words, generate_sentence

st.set_page_config(page_title="Next Word Predictor", layout="centered")

st.title("Next Word Predictor 🔮")

# Load model
model = load_model("model/model.keras")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer, max_len = pickle.load(f)

text = st.text_input("Enter text")

if st.button("Predict Next Words"):
    words = predict_next_words(model, tokenizer, text, max_len)
    st.success("Suggestions: " + ", ".join(words))

if st.button("Generate Sentence"):
    sentence = generate_sentence(model, tokenizer, text, max_len)
    st.info(sentence)