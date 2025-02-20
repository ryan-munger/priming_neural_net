import torch
import torch.nn as nn
import torch.optim as optim
import string
import random

# Define character set
letters = string.ascii_uppercase
char_to_index = {char: i for i, char in enumerate(letters)}
index_to_char = {i: char for char, i in char_to_index.items()}

# Hyperparameters
input_size = len(letters)  # 26 letters
hidden_size = 32  # Size of hidden layer
output_size = len(letters)  # Predicting one letter
num_epochs = 200
learning_rate = 0.01

# Define the RNN Model
class LetterRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LetterRNN, self).__init__()
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
    words = ['CAT', 'DOG', 'BAT', 'RAT', 'HAT', 'MAT', 'SAT']  # Small sample set
    training_data = []
    for word in words:
        x = [char_to_index[char] for char in word]
        training_data.append(x)
    return training_data

def one_hot_encode(sequence):
    tensor = torch.zeros(len(sequence), len(letters))
    for i, index in enumerate(sequence):
        tensor[i, index] = 1
    return tensor.unsqueeze(0)  # Add batch dimension

# Training the Model
model = LetterRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

training_data = generate_training_data()

for epoch in range(num_epochs):
    for sequence in training_data:
        input_seq = one_hot_encode(sequence[:-1])  # First two letters
        target = torch.tensor([sequence[-1]])  # Last letter as target
        
        hidden = model.init_hidden()
        optimizer.zero_grad()
        output, _ = model(input_seq, hidden)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the Model
def predict_missing_letter(fragment):
    input_seq = one_hot_encode([char_to_index[fragment[0]], char_to_index[fragment[2]]])
    hidden = model.init_hidden()
    output, _ = model(input_seq, hidden)
    predicted_index = torch.argmax(output).item()
    return index_to_char[predicted_index]

# Example Test Case
print("Prediction for 'B_T':", predict_missing_letter("B_T"))
