import os
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Disable oneDNN custom operations warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load trained model and tokenizer
model = tf.keras.models.load_model("word_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Function to convert a word to a sequence
def word_to_sequence(word):
    sequence = tokenizer.texts_to_sequences([word])
    return pad_sequences(sequence, maxlen=1)  # Match training maxlen=1

# Function to find valid words matching a fragment
def get_matching_words(fragment, word_list):
    fragment = fragment.lower()
    return [word for word in word_list if len(word) == 4 and all(f == "_" or f == w for f, w in zip(fragment, word))]

# Priming function (adjusts logits before softmax)
def prime_model(logits, priming_words, tokenizer):
    for word in priming_words:
        if word in tokenizer.word_index:
            word_idx = tokenizer.word_index[word]
            logits[:, word_idx] += 2.5  # Boost logits significantly before softmax - tune as needed

def main():
    with open("training-word-set.txt", "r") as file:
        words = set(line.strip().lower() for line in file.readlines())

    priming_words = input("\n\nEnter priming words separated by spaces: ").split()
    # answer fragments
    print("\n\nEnter 'exit' instead of fragment to exit.")
    while True:
        fragment = input("\nEnter Word Fragment (Ex. W_NT): ").lower()
        if fragment == "exit":
            break

        # Get valid words matching the fragment
        valid_words = get_matching_words(fragment, words)
        if not valid_words:
            print("No valid words found for this fragment.")
            continue # skip to next fragment without answering
        print(f"Possible solutions for fragment from word set: {valid_words}")

        # Get model predictions
        word_sequences = np.array([word_to_sequence(word)[0] for word in valid_words])  # Ensure shape is (batch_size, 1)
        logits = model.predict(word_sequences)

        # modifies current logits; need to re-prime every time since new logits from model.predict
        prime_model(logits, priming_words, tokenizer)

        # Select the best word based on modified logits
        best_word_index = np.argmax(logits, axis=1)[0]
        best_word = next(word for word, index in tokenizer.word_index.items() if index == best_word_index)

        print(f"Predicted word: {best_word.upper()}")

if __name__ == "__main__":
    main()