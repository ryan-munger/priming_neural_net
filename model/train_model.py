import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from collections import Counter

# Disable oneDNN custom operations warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Read words from file
with open("training-word-set.txt", "r") as file:
    words = [line.strip().lower() for line in file.readlines()]

# Count word frequencies
word_counts = Counter(words)

# Tokenize words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(words)

# Convert words to sequences
sequences = tokenizer.texts_to_sequences(words)
X = pad_sequences(sequences, maxlen=1)  # Only one token per word

# Convert targets to probability distribution
word_indices = np.array([tokenizer.word_index[word] for word in words])
y = tf.keras.utils.to_categorical(word_indices, num_classes=len(tokenizer.word_index) + 1)

# Define a simple feedforward neural network
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=1),
    Flatten(),
    Dense(len(tokenizer.word_index) + 1, activation="softmax")  # Predicts word probabilities
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X, y, epochs=50, batch_size=8, shuffle=True)  # Reducing epochs since learning is simpler

# Save the model and tokenizer
model.save("word_model.keras")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
