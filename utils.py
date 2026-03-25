import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_sequences(tokenizer, text):
    sequences = []
    for line in text.split("\n"):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            sequences.append(token_list[:i+1])
    return sequences

def preprocess_sequences(sequences):
    max_len = max(len(seq) for seq in sequences)
    sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

    X = sequences[:, :-1]
    y = sequences[:, -1]

    return X, y, max_len

def predict_next_words(model, tokenizer, text, max_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

    preds = model.predict(token_list, verbose=0)[0]
    top_indices = preds.argsort()[-3:][::-1]

    return [tokenizer.index_word[i] for i in top_indices]

def generate_sentence(model, tokenizer, text, max_len, n_words=5):
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

        preds = model.predict(token_list, verbose=0)[0]
        next_word = tokenizer.index_word[np.argmax(preds)]

        text += " " + next_word
    return text