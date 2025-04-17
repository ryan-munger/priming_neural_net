import os
import tensorflow as tf
import numpy as np
import pickle
from Levenshtein import distance
from collections import defaultdict
from scipy.special import softmax
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load models
def load_models():
    print("Loading models and data...")
    main_model = tf.keras.models.load_model("priming_model.keras")
    embedding_model = tf.keras.models.load_model("embedding_model.keras")
    
    # Load char mapping
    with open("char_mapping.pkl", "rb") as f:
        char_to_idx, max_word_length = pickle.load(f)
    
    # Load word index mapping
    with open("word_tokenizer.pkl", "rb") as f:
        word_tokenizer = pickle.load(f)
        
    # Load word set
    with open("training-word-set.txt", "r") as file:
        all_words = set(line.strip().lower() for line in file.readlines())
    
    return main_model, embedding_model, char_to_idx, max_word_length, word_tokenizer, all_words

# Convert word to input formats needed by model
def prepare_word_inputs(word, char_to_idx, max_word_length, word_tokenizer):
    # Character sequence
    char_seq = np.zeros((1, max_word_length))
    for i, char in enumerate(word.lower()):
        if i < max_word_length and char in char_to_idx:
            char_seq[0, i] = char_to_idx[char]
    
    # Word sequence (if word is in vocabulary)
    if word.lower() in word_tokenizer.word_index:
        word_seq = np.array([[word_tokenizer.word_index[word.lower()]]])
    else:
        # Handle OOV words by using a placeholder
        word_seq = np.array([[0]])  # Use 0 as padding/OOV token
    
    return word_seq, char_seq

# Find valid words matching a fragment
def get_matching_words(fragment, word_list):
    fragment = fragment.lower()
    return [word for word in word_list if len(word) == len(fragment) and 
            all(f == "_" or f == w for f, w in zip(fragment, word))]

# Calculate orthographic similarity between words
def get_orthographic_similarity(word1, word2):
    max_len = max(len(word1), len(word2))
    return 1 - (distance(word1, word2) / max_len)

# Find orthographic neighbors
def find_orthographic_neighbors(word, word_list, threshold=0.6):
    neighbors = []
    for candidate in word_list:
        if candidate != word:
            sim = get_orthographic_similarity(word, candidate)
            if sim >= threshold:
                neighbors.append((candidate, sim))
    return sorted(neighbors, key=lambda x: x[1], reverse=True)

# Calculate embedding similarity between two words
def get_embedding_similarity(word1_emb, word2_emb):
    # Cosine similarity
    return np.dot(word1_emb, word2_emb) / (np.linalg.norm(word1_emb) * np.linalg.norm(word2_emb))

# Apply priming using the embeddings from the model
def prime_with_embeddings(priming_words, target_words, embedding_model, char_to_idx, 
                          max_word_length, word_tokenizer):
    # Get embeddings for priming words
    prime_embeddings = {}
    for word in priming_words:
        word_seq, char_seq = prepare_word_inputs(word, char_to_idx, max_word_length, word_tokenizer)
        # Get embedding from the bottleneck layer
        embedding = embedding_model.predict([word_seq, char_seq], verbose=0)[0]
        prime_embeddings[word] = embedding / np.linalg.norm(embedding)  # Normalize
    
    # Get embeddings for target words
    target_embeddings = {}
    for word in target_words:
        word_seq, char_seq = prepare_word_inputs(word, char_to_idx, max_word_length, word_tokenizer)
        embedding = embedding_model.predict([word_seq, char_seq], verbose=0)[0]
        target_embeddings[word] = embedding / np.linalg.norm(embedding)  # Normalize
    
    # Calculate priming effects 
    priming_effects = {}
    for target_word in target_words:
        # Calculate similarity in the embedding space
        embedding_effect = 0
        for prime_word in priming_words:
            similarity = get_embedding_similarity(prime_embeddings[prime_word], 
                                                target_embeddings[target_word])
            embedding_effect = max(embedding_effect, similarity)

        # Orthographic priming (visual form similarity)
        ortho_effect = 0
        for prime_word in priming_words:
            ortho_sim = get_orthographic_similarity(prime_word, target_word)
            ortho_effect = max(ortho_effect, ortho_sim)

        # Combined effect (combining neural and visual similarity)
        combined_effect = (0.4 * embedding_effect) + (0.6 * ortho_effect)
        
        # Direct match gets maximum boost
        if target_word in priming_words:
            priming_effects[target_word] = 2.0  # Direct prime
        else:
            # Scale the effect
            priming_effects[target_word] = combined_effect * 1.5
    
    return priming_effects

# Apply priming boosts to logits
def apply_priming_to_logits(logits, priming_effects, word_tokenizer):
    # Create a copy to avoid modifying the original
    primed_logits = logits.copy()
    
    # Apply priming boosts
    for word, boost in priming_effects.items():
        if word in word_tokenizer.word_index:
            word_idx = word_tokenizer.word_index[word]
            primed_logits[0, word_idx] += boost
    
    return primed_logits


def main():
    # Load models and data
    main_model, embedding_model, char_to_idx, max_word_length, word_tokenizer, all_words = load_models()
    
    # Get priming words
    priming_words = input("\n\nEnter priming words separated by spaces: ").lower().split()
    
    # Print orthographic neighbors of priming words
    print("\nOrthographic neighbors of priming words:")
    for word in priming_words:
        neighbors = find_orthographic_neighbors(word, all_words)[:5]  # Top 5 neighbors
        if neighbors:
            neighbor_str = ", ".join([f"{w} ({s:.2f})" for w, s in neighbors])
            print(f"  {word}: {neighbor_str}")
        else:
            print(f"  {word}: No close neighbors found")
    
    print("\nEnter 'exit' to stop.")
    while True:
        fragment = input("\nEnter Word Fragment (e.g., W_NT): ").lower()
        if fragment == "exit":
            break
        
        # Find valid words matching the fragment
        valid_words = get_matching_words(fragment, all_words)
        if not valid_words:
            print("No valid words found for this fragment.")
            continue
        
        print(f"Possible completions: {', '.join(valid_words)}")
        
        # Prepare inputs for all valid words
        word_inputs = []
        char_inputs = []
        
        for word in valid_words:
            word_seq, char_seq = prepare_word_inputs(word, char_to_idx, max_word_length, word_tokenizer)
            word_inputs.append(word_seq)
            char_inputs.append(char_seq)
        
        # Concatenate inputs
        word_inputs = np.vstack(word_inputs)
        char_inputs = np.vstack(char_inputs)
        
        # Get model predictions without priming
        logits = main_model.predict([word_inputs, char_inputs], verbose=0)
        
        # Calculate baseline probabilities
        print("\nProbabilities before priming:")
        probs_before = softmax(logits, axis=1)
        
        # Calculate total probability mass for valid words only
        total_valid_prob = sum(probs_before[i, idx] for i, word in enumerate(valid_words) 
                              for idx in [word_tokenizer.word_index[word]])
        
        for i, word in enumerate(valid_words):
            idx = word_tokenizer.word_index[word]
            raw_prob = probs_before[i, idx]
            normalized_prob = (raw_prob / total_valid_prob) * 100
            print(f"{word:>5}: {normalized_prob:.2f}%")
        
        # Apply priming
        priming_effects = prime_with_embeddings(
            priming_words, valid_words, embedding_model, char_to_idx, 
            max_word_length, word_tokenizer
        )
        
        # Print priming effects
        print("\nPriming effects on possible completions:")
        for word, effect in priming_effects.items():
            source = "(direct prime)" if word in priming_words else "(spreading activation)"
            print(f"  {word}: +{effect:.2f} {source}")
        
        # Apply priming to logits
        primed_logits = np.array([apply_priming_to_logits(logits[i:i+1], priming_effects, 
                                                         word_tokenizer)[0] 
                                 for i in range(len(valid_words))])
        
        # Calculate primed probabilities
        print("\nProbabilities after priming:")
        probs_after = softmax(primed_logits, axis=1)
        
        # Calculate total probability mass for valid words only
        total_valid_prob_after = sum(probs_after[i, idx] for i, word in enumerate(valid_words) 
                                   for idx in [word_tokenizer.word_index[word]])
        
        for i, word in enumerate(valid_words):
            idx = word_tokenizer.word_index[word]
            raw_prob = probs_after[i, idx]
            normalized_prob = (raw_prob / total_valid_prob_after) * 100
            print(f"{word:>5}: {normalized_prob:.2f}%")
        
        # Get most likely word
        best_indices = np.argmax(primed_logits, axis=1)
        best_probs = [probs_after[i, idx] for i, idx in enumerate(best_indices)]
        best_word_idx = np.argmax(best_probs)
        best_word = valid_words[best_word_idx]
        
        print(f"\nPredicted word: {best_word.upper()}")
        
        # Orthographic similarity report for debug
        print("\nOrthographic similarity to priming words:")
        for word in valid_words:
            similarities = [f"{pw}: {get_orthographic_similarity(word, pw):.2f}" for pw in priming_words]
            is_predicted = " (PREDICTED)" if word == best_word else ""
            print(f"  {word}{is_predicted}: {', '.join(similarities)}")

if __name__ == "__main__":
    main()