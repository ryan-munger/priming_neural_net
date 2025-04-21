import os
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt

def getWordSet(path: str):
    with open(path, "r") as file:
        return [line.strip().lower() for line in file.readlines()]

# for orthography
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

# Generate context word pairs for skip-gram model
def generate_context_pairs(words, window_size=2):
    word_pairs = []
    word_to_idx = {word: i for i, word in enumerate(set(words))}
    
    for i, target_word in enumerate(words):
        # Get context words within window
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:  # Skip the target word itself
                context_word = words[j]
                word_pairs.append((target_word, context_word))
    
    return word_pairs, word_to_idx

def main():
    # Disable oneDNN custom operations warning
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

    # Load word training set
    words = getWordSet("training-word-set.txt")
    
    # Count word frequencies in set
    word_counts = Counter(words)
    print(f"Total unique words: {len(set(words))}")
    
    # Tokenize words
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(words)
    
    # Create character-level representations
    char_sequences, char_to_idx, max_word_length = create_character_features(words)
    
    # Encode all words as sequences
    word_sequences = word_tokenizer.texts_to_sequences(words)
    
    # Create an autoencoder model to learn frequency-weighted word representations
    # We do NOT want to be predicting the 'next' word
    
    # Input layer - one-hot encoded word and character sequence
    word_input = Input(shape=(1,), name='word_input')
    char_input = Input(shape=(max_word_length,), name='char_input')
    
    # Word embedding layer - this will become our representation
    embedding_dim = 32  # Increase embedding dimension for better representation
    word_embedding = Embedding(input_dim=len(word_tokenizer.word_index) + 1, 
                              output_dim=embedding_dim, 
                              name='word_embedding')(word_input)
    word_embedding = Flatten()(word_embedding)
    
    # Character embedding - to capture orthographic features
    char_embedding = Embedding(input_dim=len(char_to_idx) + 1, 
                              output_dim=16)(char_input)
    char_features = LSTM(24)(char_embedding)
    
    # Combine both feature sets into one representation
    combined = tf.keras.layers.concatenate([word_embedding, char_features])
    encoded = Dense(64, activation='relu')(combined)
    encoded = Dropout(0.2)(encoded)
    
    # Bottleneck layer - frequency-weighted representation
    bottleneck = Dense(24, activation='relu', name='frequency_embedding')(encoded)
    
    # Decoder - reconstruct the word identity
    decoded = Dense(64, activation='relu')(bottleneck)
    decoded = Dropout(0.2)(decoded)
    output = Dense(len(word_tokenizer.word_index) + 1, activation='softmax')(decoded)
    
    # Create model
    model = tf.keras.models.Model(inputs=[word_input, char_input], outputs=output)
    
    # Compile with standard categorical crossentropy
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Create one-hot encoded targets
    y = tf.keras.utils.to_categorical([word_tokenizer.word_index[word] for word in words], 
                                     num_classes=len(word_tokenizer.word_index) + 1)
    
    # Create sample weights directly proportional to word frequency
    # This is key - lets more frequent words have higher weights
    sample_weights = np.array([word_counts[word] for word in words], dtype=np.float32)
    
    # Apply log transformation to compress the range of frequencies
    sample_weights = np.log1p(sample_weights)  # log(1+x) to handle counts of 1
    sample_weights /= sample_weights.max()  # Normalize to [0,1]
    
    # Scale weights more aggressively to emphasize frequency differences
    sample_weights = sample_weights ** 2  # Square the weights to emphasize high frequency words
    
    # Train the model
    history = model.fit(
        [pad_sequences(word_sequences, maxlen=1), char_sequences], 
        y, 
        epochs=50, # 50 should be fine
        batch_size=32,
        sample_weight=sample_weights,  # Apply frequency-based weights
        shuffle=True,
        verbose=1
    )
    
    # Plot training history for debug
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'])
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('training_loss.png')
    plt.close()
    
    # Save the full model
    model.save("priming_model.keras")
    
    # Create a separate embedding extractor model
    embedding_model = tf.keras.models.Model(
        inputs=[word_input, char_input],
        outputs=model.get_layer('frequency_embedding').output
    )
    embedding_model.save("embedding_model.keras")
    
    # Save tokenizers and character mapping
    with open("word_tokenizer.pkl", "wb") as f:
        pickle.dump(word_tokenizer, f)
    
    with open("char_mapping.pkl", "wb") as f:
        pickle.dump((char_to_idx, max_word_length), f)
    
    # Analyze and visualize word embeddings by frequency
    print("\nAnalyzing word embedding strength...")
    
    # Get all unique words and their frequencies
    unique_words = list(set(words))
    
    # Create inputs for the embedding model
    X_word_test = pad_sequences(word_tokenizer.texts_to_sequences(unique_words), maxlen=1)
    X_char_test = np.zeros((len(unique_words), max_word_length))
    
    for i, word in enumerate(unique_words):
        for j, char in enumerate(word):
            if j < max_word_length:
                X_char_test[i, j] = char_to_idx[char]
    
    # Get embeddings for all words
    embeddings = embedding_model.predict([X_word_test, X_char_test])
    
    # Calculate embedding norm (magnitude) as a measure of representation strength
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    
    # Create frequency-norm pairs for analysis
    word_freq_norm = [(word, word_counts[word], norm) for word, norm in zip(unique_words, embedding_norms)]
    
    # Sort by frequency
    word_freq_norm.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 10 and bottom 10 words by frequency and their embedding norms
    print("\nTop 10 frequency words and their embedding strengths:")
    for word, freq, norm in word_freq_norm[:10]:
        print(f"Word: {word}, Frequency: {freq}, Embedding strength: {norm:.4f}")
    
    print("\nBottom 10 frequency words and their embedding strengths:")
    for word, freq, norm in word_freq_norm[-10:]:
        print(f"Word: {word}, Frequency: {freq}, Embedding strength: {norm:.4f}")
    
    # Calculate average norm by frequency quartile
    word_freq_norm.sort(key=lambda x: x[1])  # Sort by ascending frequency
    quartile_size = len(word_freq_norm) // 4
    
    quartiles = [
        word_freq_norm[:quartile_size],                      # Q1 (lowest freq)
        word_freq_norm[quartile_size:2*quartile_size],       # Q2
        word_freq_norm[2*quartile_size:3*quartile_size],     # Q3
        word_freq_norm[3*quartile_size:],                    # Q4 (highest freq)
    ]
    
    print("\nAverage embedding strength by frequency quartile:")
    for i, quartile in enumerate(quartiles):
        avg_norm = sum(item[2] for item in quartile) / len(quartile)
        avg_freq = sum(item[1] for item in quartile) / len(quartile)
        print(f"Quartile {i+1}: Avg frequency: {avg_freq:.2f}, Avg embedding strength: {avg_norm:.4f}")
    
    # Create frequency vs. embedding strength scatter plot
    frequencies = [item[1] for item in word_freq_norm]
    norms = [item[2] for item in word_freq_norm]

    plt.figure(figsize=(10, 6))
    plt.scatter(frequencies, norms, alpha=0.5)
    plt.title("Word Frequency vs. Embedding Strength", fontsize=18, fontweight='bold')
    plt.xlabel("Word Frequency", fontsize=20, fontweight='bold')
    plt.ylabel("Embedding Strength (Norm)", fontsize=20, fontweight='bold')

    plt.xscale('log')  # Log scale for better visualization

    # Add trend line
    z = np.polyfit(np.log(frequencies), norms, 1)
    p = np.poly1d(z)
    plt.plot(sorted(frequencies), p(np.log(sorted(frequencies))), "r--", alpha=0.8)

    plt.tight_layout()
    plt.savefig("frequency_vs_strength.png")
    plt.close()


if __name__ == "__main__":
    main()
