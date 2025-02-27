def extract_four_letter_words(input_file, output_file):
  try:
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
      for line in infile:
        parts = line.split()
        if len(parts) >= 2:  # ensure word and number
          word = parts[0]
          if len(word) == 4:
            outfile.write(line) # write back word and number

  except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
  except Exception as e:
    print(f"An error occurred: {e}")

if __name__ == "__main__":
  inputFile = "third_of_million_words_freq.txt"
  outputFile = "common_4_letter_words.txt"
  extract_four_letter_words(inputFile, outputFile)

  print(f"Four-letter words (with counts) extracted and written to '{outputFile}'.")