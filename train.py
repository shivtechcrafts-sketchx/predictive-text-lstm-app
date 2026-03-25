import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from utils import create_sequences, preprocess_sequences

# Load data
with open("data/text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Sequences
sequences = create_sequences(tokenizer, text)
X, y, max_len = preprocess_sequences(sequences)

y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Model
model = Sequential([
    Embedding(total_words, 50),
    LSTM(150, return_sequences=True),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=200, verbose=1)

# Save
model.save("model/model.keras")

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump((tokenizer, max_len), f)

print("✅ Training complete")