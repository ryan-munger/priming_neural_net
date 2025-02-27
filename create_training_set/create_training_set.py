import random

def generate_word_list(input_file, output_list_size=5000):
  word_frequencies = {}
  total_frequency = 0

  try:
    with open(input_file, 'r') as infile:
      for line in infile:
        parts = line.split()
        if len(parts) == 2:
          word, frequency = parts[0], int(parts[1])
          word_frequencies[word] = frequency
          total_frequency += frequency

      if not word_frequencies:
        return []  

      # add words in amt based on their freq
      output_list = []
      for word, frequency in word_frequencies.items():
        count = round((frequency / total_frequency) * output_list_size)
        output_list.extend([word] * count)

      # If the list is shorter than the desired size, add random words from the available pool.
      while len(output_list) < output_list_size:
        random_word = random.choice(list(word_frequencies.keys()))
        output_list.append(random_word)

      # If the list is longer than the desired size, remove random words.
      while len(output_list) > output_list_size:
        random_index = random.randint(0, len(output_list) - 1)
        output_list.pop(random_index)

      random.shuffle(output_list) #shuffle the list so is not in freq order
      return output_list

  except FileNotFoundError:
      print(f"Error: Input file '{input_file}' not found.")
      return []
  except ValueError:
      print("Error: Invalid frequency value in the input file.")
      return []
  except Exception as e:
      print(f"An error occurred: {e}")
      return []


if __name__ == "__main__":
  input_filename = "words_750.txt" 
  output_list = generate_word_list(input_filename)

  print(f"Generated a list of {len(output_list)} words.")
  with open("training_words.txt", "w") as outfile:
      for word in output_list:
          outfile.write(word + "\n")
