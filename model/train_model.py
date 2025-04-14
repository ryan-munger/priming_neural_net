import os
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from collections import Counter

def getWordSet(path: str):
    with open(path, "r") as file:
        return [line.strip().lower() for line in file.readlines()]

def create_character_features(words):
    # Create character-level encodings
    all_chars = set(''.join(words))
    char_to_idx = {c: i+1 for i, c in enumerate(sorted(all_chars))}  # 0 reserved for padding
    
    # Convert words to character sequences
    max_word_length = max(len(word) for word in words)
    char_sequences = np.zeros((len(words), max_word_length))
    
    for i, word in enumerate(words):
        for j, char in enumerate(word):
            char_sequences[i, j] = char_to_idx[char]
    
    return char_sequences, char_to_idx, max_word_length

def main():
    # Disable oneDNN custom operations warning
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

    words = getWordSet("training-word-set.txt")

    # Count word frequencies
    word_counts = Counter(words)

    # Tokenize words for word-level representation
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(words)
    
    # Create character-level representations for orthographic features
    char_sequences, char_to_idx, max_word_length = create_character_features(words)
    
    # Create combined model that learns from both word identity and character patterns
    
    # Word identity branch
    word_sequences = word_tokenizer.texts_to_sequences(words)
    X_word = pad_sequences(word_sequences, maxlen=1)
    
    # Character sequences branch
    X_char = char_sequences
    
    # Define the model with two input branches
    word_input = Input(shape=(1,))
    char_input = Input(shape=(max_word_length,))
    
    # Word embedding branch
    word_embedding = Embedding(input_dim=len(word_tokenizer.word_index) + 1, output_dim=8)(word_input)
    word_embedding = Flatten()(word_embedding)
    
    # Character embedding branch - captures orthographic patterns
    char_embedding = Embedding(input_dim=len(char_to_idx) + 1, output_dim=4)(char_input)
    char_features = LSTM(16)(char_embedding)  # Process letter sequences
    
    # Combine both feature sets
    combined = tf.keras.layers.concatenate([word_embedding, char_features])
    
    # Output layer
    output = Dense(len(word_tokenizer.word_index) + 1, activation="softmax")(combined)
    
    # Create and compile model
    model = tf.keras.models.Model(inputs=[word_input, char_input], outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Convert targets to categorical labels
    y = tf.keras.utils.to_categorical([word_tokenizer.word_index[word] for word in words], 
                                     num_classes=len(word_tokenizer.word_index) + 1)
    
    # Sample weights based on word frequency
    sample_weights = np.array([word_counts[word] for word in words], dtype=np.float32)
    sample_weights /= sample_weights.max()  # Normalize weights to [0,1] range
    
    # Train the model
    model.fit([X_word, X_char], y, epochs=50, batch_size=8, 
             sample_weight=sample_weights, shuffle=True)
    
    # Save the model
    model.save("priming_model.keras")
    
    # Save tokenizers and character mapping
    with open("word_tokenizer.pkl", "wb") as f:
        pickle.dump(word_tokenizer, f)
    
    with open("char_mapping.pkl", "wb") as f:
        pickle.dump((char_to_idx, max_word_length), f)

if __name__ == "__main__":
    main()