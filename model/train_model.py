import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from collections import Counter

def getWordSet(path: str):
    with open(path, "r") as file:
        return [line.strip().lower() for line in file.readlines()]

def main():
    # Disable oneDNN custom operations warning
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    words = getWordSet("training-word-set.txt")

    # Count word frequencies
    word_counts = Counter(words)

    # Tokenize words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words)

    # Convert words to sequences
    sequences = tokenizer.texts_to_sequences(words)
    X = pad_sequences(sequences, maxlen=1)  # Only one token per word

    # Convert targets to categorical labels
    y = tf.keras.utils.to_categorical([tokenizer.word_index[word] for word in words], num_classes=len(tokenizer.word_index) + 1)

    # Create sample weights based on word frequency
    sample_weights = np.array([word_counts[word] for word in words], dtype=np.float32)
    sample_weights /= sample_weights.max()  # Normalize weights to [0,1] range

    # Define a simple feedforward neural network
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=1),
        Flatten(),
        Dense(len(tokenizer.word_index) + 1, activation="softmax")  # Predicts word probabilities
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model with sample weighting
    model.fit(X, y, epochs=50, batch_size=8, sample_weight=sample_weights, shuffle=True)

    # Save the model and tokenizer
    model.save("word_model.keras")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

if __name__ == "__main__":
    main()