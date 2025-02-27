import torch
import torch.nn as nn
import torch.optim as optim
import string
import random

# Define character set
letters = string.ascii_uppercase
char_to_index = {char: i for i, char in enumerate(letters)}
index_to_char = {i: char for char, i in char_to_index.items()}

def read_word_list(filename): 
    word_list = []
    try:
        with open(filename, 'r') as infile:
            for line in infile:
                word = line.strip()  # Remove leading/trailing whitespace (including newline)
                if word:  # Check if the line is not empty
                    word_list.append(word.upper())
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return word_list

# Define word encoding dictionary
word_to_index = {}
index_to_word = {}
word_list = read_word_list("training_words.txt")
# print(word_list)
for i, word in enumerate(word_list):
    word_to_index[word] = i
    index_to_word[i] = word

# Hyperparameters
input_size = len(letters)  # 26 letters
hidden_size = 128  # Increased hidden size for more complex patterns
output_size = len(word_list)  # Predicting one word
num_epochs = 500
learning_rate = 0.001

# Define the RNN Model
class WordRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WordRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Take last time step output
        return out, hidden
    
    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

# Generate training data
def generate_training_data():
    training_data = []
    for word in word_list:
        x = [char_to_index[char] for char in word]
        training_data.append((x, word_to_index[word]))
    return training_data

def one_hot_encode(sequence):
    tensor = torch.zeros(len(sequence), len(letters))
    for i, index in enumerate(sequence):
        tensor[i, index] = 1
    return tensor.unsqueeze(0)  # Add batch dimension

# Training the Model
model = WordRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

training_data = generate_training_data()

for epoch in range(num_epochs):
    for sequence, target in training_data:
        input_seq = one_hot_encode(sequence[:-1])  # First three letters
        target_tensor = torch.tensor([target])  # Target word index
        
        hidden = model.init_hidden()
        optimizer.zero_grad()
        output, _ = model(input_seq, hidden)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the Model
def predict_missing_letter(fragment):
    # Extract known letters, ignoring the missing slot
    known_indices = [char_to_index[fragment[i]] for i in range(4) if fragment[i] != '_']
    
    # Ensure input length is correct by padding with a default letter (e.g., 'A')
    while len(known_indices) < 3:
        known_indices.append(char_to_index['A'])  # Neutral padding
    
    input_seq = one_hot_encode(known_indices)
    
    hidden = model.init_hidden()
    output, _ = model(input_seq, hidden)
    predicted_index = torch.argmax(output).item()
    
    return index_to_word[predicted_index]


# Example Test Case
print("Prediction for 'W_NT':", predict_missing_letter("W_NT"))
print("Prediction for 'S_NG':", predict_missing_letter("S_NG"))
print("Prediction for 'Q_IT':", predict_missing_letter("Q_IT"))
