import os
import tensorflow as tf
import numpy as np
import pickle
from Levenshtein import distance
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
from scipy.special import softmax

# Disable oneDNN custom operations warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load trained model and tokenizer
model = tf.keras.models.load_model("priming_model.keras")
with open("word_tokenizer.pkl", "rb") as f:
    word_tokenizer = pickle.load(f)

with open("char_mapping.pkl", "rb") as f:
    char_to_idx, max_word_length = pickle.load(f)

# Create character-level sequences
def word_to_char_sequence(word):
    char_seq = np.zeros((1, max_word_length))
    for i, char in enumerate(word):
        if i < max_word_length and char in char_to_idx:
            char_seq[0, i] = char_to_idx[char]
    return char_seq

# Convert a word to sequences (both word and char level)
def word_to_sequences(word):
    word_sequence = word_tokenizer.texts_to_sequences([word])
    word_sequence = pad_sequences(word_sequence, maxlen=1)
    char_sequence = word_to_char_sequence(word)
    return word_sequence, char_sequence

# Find valid words matching a fragment
def get_matching_words(fragment, word_list):
    fragment = fragment.lower()
    return [word for word in word_list if len(word) == len(fragment) and 
            all(f == "_" or f == w for f, w in zip(fragment, word))]

# Orthographic similarity score
def get_orthographic_similarity(word1, word2):
    # Levenshtein distance to measure similarity
    max_len = max(len(word1), len(word2))
    return 1 - (distance(word1, word2) / max_len)  # Higher value: more similar

# Find orthographic neighbors
def find_orthographic_neighbors(word, word_list, threshold=0.6):
    neighbors = []
    for candidate in word_list:
        if candidate != word:  # Don't include the word itself lol
            similarity = get_orthographic_similarity(word, candidate)
            if similarity >= threshold:
                neighbors.append((candidate, similarity))
    return neighbors

# Prime identity strongly (word itself) and prime using proportional orthographic spread
def prime_model_with_spread(logits, priming_words, tokenizer, all_words):
    # Build a similarity network for activation spreading
    similarity_network = defaultdict(list)
    
    # Level 1: direct priming of explicitly mentioned word identitiess
    directly_primed_words = set(priming_words)
    
    # level 2: find orthographic neighbors of primed words for priming
    for primed_word in priming_words:
        neighbors = find_orthographic_neighbors(primed_word, all_words)
        for neighbor, similarity in neighbors:
            similarity_network[primed_word].append((neighbor, similarity))
            # Add to the set of words receiving some priming
            if similarity > 0.6:  # Threshold for significant enough similarity
                directly_primed_words.add(neighbor)
    
    # Apply logit priming boost
    for word in directly_primed_words:
        if word in tokenizer.word_index:
            word_idx = tokenizer.word_index[word]
            # Directly primed words receive the full boost
            if word in priming_words:
                logits[:, word_idx] += 2.0 # TUNE THIS
            # Orthographic neighbors get a smaller boost based on similarity
            else:
                # Find max similarity to any priming word
                max_similarity = max(get_orthographic_similarity(word, pw) 
                                    for pw in priming_words)
                logits[:, word_idx] += max_similarity * 1.2
    
    # Activation spread info for user
    print("\nPriming activation spread:")
    for word in directly_primed_words:
        if word in priming_words:
            print(f"  {word}: +2.0 (direct prime)")
        else:
            max_sim = max(get_orthographic_similarity(word, pw) for pw in priming_words)
            print(f"  {word}: +{max_sim * 1.2:.2f} (orthographic neighbor)")

def main():
    with open("training-word-set.txt", "r") as file:
        all_words = set(line.strip().lower() for line in file.readlines())

    priming_words = input("\n\nEnter priming words separated by spaces: ").split()
    priming_words = [word.lower() for word in priming_words] # make them lower
    
    # Show orthographic neighbors of priming words
    print("\nOrthographic neighbors of priming words:")
    for word in priming_words:
        neighbors = find_orthographic_neighbors(word, all_words)
        neighbors.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity
        top_neighbors = neighbors[:5]  # Show top 5 neighbors
        if top_neighbors:
            neighbor_str = ", ".join([f"{w} ({s:.2f})" for w, s in top_neighbors])
            print(f"  {word}: {neighbor_str}")
        else:
            print(f"  {word}: No close neighbors found")
    
    # complete word fragments
    print("\n\nEnter 'exit' instead of a fragment to exit.")
    while True:
        fragment = input("\nEnter Word Fragment (Ex. W_NT): ").lower()
        if fragment == "exit":
            break

        # Find possible solutions
        valid_words = get_matching_words(fragment, all_words)
        if not valid_words:
            print("No valid words found for this fragment.")
            continue # skip to next fragment - careful to keep fragments within wordset
        print(f"Possible solutions for fragment from word set: {valid_words}")
        
        # Prepare batch inputs for prediction
        word_sequences = []
        char_sequences = []
        
        # We see which of the valid solutions the model likes the best
        for word in valid_words:
            w_seq, c_seq = word_to_sequences(word)
            word_sequences.append(w_seq[0])
            char_sequences.append(c_seq[0])
        
        # Convert to numpy arrays
        word_sequences = np.array(word_sequences)
        char_sequences = np.array(char_sequences)
        
        # Get model predictions 
        logits = model.predict([word_sequences, char_sequences])

        # Print the probabilities of each valid word (strength) before priming
        print("Strength of valid words before priming effects:")
        sum_valid_prob_before = 0
        probs_before = softmax(logits[0])
        for word in valid_words:
            idx = word_tokenizer.word_index[word]
            print(f"{word:>5}: prob={probs_before[idx]:.10f}")
            sum_valid_prob_before += probs_before[idx]

        print("\nProbability of valid word selection before priming effects:")
        for word in valid_words:
            idx = word_tokenizer.word_index[word]
            pct = (probs_before[idx] / sum_valid_prob_before) * 100
            print(f"{word:>5}: prob={pct:.4f}%")

        
        valid_word_indices = [word_tokenizer.word_index[word] for word in valid_words 
                             if word in word_tokenizer.word_index]

        # Prime the model with orthographic spread activation
        prime_model_with_spread(logits, priming_words, word_tokenizer, all_words)

        # Print the probabilities of each valid word (strength) after priming
        print("\nStrength of valid words after priming effects:")
        sum_valid_prob_after = 0
        probs_after = softmax(logits[0])
        for word in valid_words:
            idx = word_tokenizer.word_index[word]
            print(f"{word:>5}: prob={probs_after[idx]:.10f}")
            sum_valid_prob_after += probs_after[idx]

        print("\nProbability of valid word selection after priming effects:")
        for word in valid_words:
            idx = word_tokenizer.word_index[word]
            pct = (probs_after[idx] / sum_valid_prob_after) * 100
            print(f"{word:>5}: prob={pct:.4f}%")

        # Mask logits so that only valid solutions are considered 
        masked_logits = np.full(logits.shape, -np.inf)
        masked_logits[:, valid_word_indices] = logits[:, valid_word_indices]

        # Select the best solution word from the valid set of solutions
        best_word_index = np.argmax(masked_logits, axis=1)[0]
        best_word = next(word for word, index in word_tokenizer.word_index.items() 
                        if index == best_word_index)

        print(f"\nPredicted word: {best_word.upper()}")
        
        # orthographic similarity analysis for the fragment completions for debug
        print("\nOrthographic similarity of possible completions to priming words:")
        for word in valid_words:
            similarities = []
            for primed_word in priming_words:
                sim = get_orthographic_similarity(word, primed_word)
                similarities.append(f"{primed_word}: {sim:.2f}")
            
            is_predicted = " (PREDICTED)" if word == best_word else ""
            print(f"  {word}{is_predicted}: {', '.join(similarities)}")

if __name__ == "__main__":
    main()