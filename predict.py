import pickle
from tensorflow.keras.models import load_model
from utils import predict_next_word

# Load model
model = load_model("model/model.h5")

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer, max_seq_len = pickle.load(f)

# Test
while True:
    text = input("Enter text: ")
    print("Next word:", predict_next_word(model, tokenizer, text, max_seq_len))