import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Inputs
fragment = input("Fragment: ")
word_a = input("Word A: ")
word_b = input("Word B: ")

# Hardcoded :(
data = {
    'HUMANS': {
        'A': {'Other': 0.35, word_a: 0.59, word_b: 0.06},
        'B': {'Other': 0.57, word_a: 0.12, word_b: 0.31},
        'CONTROL': {'Other': 0.78, word_a: 0.13, word_b: 0.09}
    },
    'MODEL': {
        'A': {'Other': 0.33, word_a: 0.59, word_b: 0.08},
        'B': {'Other': 0.37, word_a: 0.11, word_b: 0.52},
        'CONTROL': {'Other': 0.70, word_a: 0.16, word_b: 0.14}
    }
}

conditions = ['A', 'B', 'CONTROL']
response_types = ['Other', word_a, word_b]
group_labels = ['HUMANS', 'MODEL']
colors = ['lightgray', 'salmon', 'royalblue']  

# Prep data for plotting
human_data = np.array([[data['HUMANS'][cond][resp] for resp in response_types] for cond in conditions])
model_data = np.array([[data['MODEL'][cond][resp] for resp in response_types] for cond in conditions])

bar_width = 0.35

# Set positions bars on the x
r1 = np.arange(len(conditions))
r2 = [x + bar_width for x in r1]

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Plot stacked bars for HUMANS
bottom_human = np.zeros(len(conditions))
for i, response in enumerate(response_types):
    ax.bar(r1, human_data[:, i], bar_width, bottom=bottom_human, color=colors[i], label=response)
    bottom_human += human_data[:, i]

# Plot stacked bars for MODEL
bottom_model = np.zeros(len(conditions))
for i, response in enumerate(response_types):
    ax.bar(r2, model_data[:, i], bar_width, bottom=bottom_model, color=colors[i])
    bottom_model += model_data[:, i]

# Add labels and title
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Rate of Selection', fontweight='bold')
ax.set_title('Humans Vs Model: ' + fragment, fontweight='bold')
ax.set_xticks([r + bar_width/2 for r in range(len(conditions))])
ax.set_xticklabels(conditions)

# legendary code
ax.legend(title='Response Type')

# Add vertical group labels on the bars themselves; credit: ChatGPT
for label, x_locs, data_array in zip(group_labels, [r1, r2], [human_data, model_data]):
    for x, heights in zip(x_locs, data_array):
        total_height = sum(heights)
        ax.text(x, total_height / 2, label, ha='center', va='center', rotation='vertical', fontweight='bold', color='white', fontsize=10)

# Adjust layout
plt.tight_layout()
plt.show()
