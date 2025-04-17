import random
import math
import matplotlib.pyplot as plt
from collections import Counter

# while frequency is important, 'that' is 90,000x more frequent than 'sool'
# therefore, we need to do some scaling
def generate_word_list_power_scaled(input_file, output_list_size=10000, alpha=0.75):
    word_frequencies = {}
    total_frequency = 0

    try:
        # Read input file
        with open(input_file, 'r') as infile:
            for line in infile:
                parts = line.strip().split()
                if len(parts) == 2:
                    word, frequency = parts[0], int(parts[1])
                    word_frequencies[word] = frequency
                    total_frequency += frequency

        if not word_frequencies:
            return []

        # Scale frequencies with alpha power
        adjusted_frequencies = {w: f ** alpha for w, f in word_frequencies.items()}
        total_adjusted = sum(adjusted_frequencies.values())

        # Assign at least 1 per word, then distribute the rest
        base_list = list(word_frequencies.keys())
        remaining_slots = output_list_size - len(base_list)

        # Normalize adjusted frequencies to distribute remaining slots
        normalized = {
            word: adjusted_frequencies[word] / total_adjusted
            for word in word_frequencies
        }

        # Distribute remaining slots based on normalized adjusted frequencies
        extra_counts = {
            word: int(normalized[word] * remaining_slots)
            for word in word_frequencies
        }

        # Add guaranteed one + extras
        word_list = []
        for word in word_frequencies:
            count = 1 + extra_counts[word]
            word_list.extend([word] * count)

        # Fix length if it's off due to rounding
        if len(word_list) > output_list_size:
            word_list = random.sample(word_list, output_list_size)
        elif len(word_list) < output_list_size:
            extra_needed = output_list_size - len(word_list)
            extra_words = random.choices(list(word_frequencies.keys()), k=extra_needed)
            word_list.extend(extra_words)

        random.shuffle(word_list)
        return word_list

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return []
    except ValueError:
        print("Error: Invalid frequency value in the input file.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Histogram plot
def plot_histogram(word_list, top_n=50):
    counter = Counter(word_list)
    most_common = counter.most_common(top_n)
    words, counts = zip(*most_common)

    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.xticks(rotation=90)
    plt.title(f"Top {top_n} Words in Output List")
    plt.xlabel("Words")
    plt.ylabel("Occurrences")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_filename = "words_750.txt"
    output_list_size = 10000

    output_list = generate_word_list_power_scaled(input_filename, output_list_size=output_list_size)

    print(f"Generated a word list of {len(output_list)} words.")
    with open("training_words_power_scaled.txt", "w") as outfile:
        for word in output_list:
            outfile.write(word + "\n")

    # Plot histogram
    plot_histogram(output_list)
