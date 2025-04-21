import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Inputs
fragment = input("Fragment: ")
word_a = input("Word A: ")
word_b = input("Word B: ")

# Hardcoded :(
data = { # W_NT
    'HUMANS': {
        'A': {'Other': 0.06, word_a: 0.54, word_b: 0.40, 'Total': 1},
        'B': {'Other': 0.01, word_a: 0.06, word_b: 0.93, 'Total': 1},
        'CONTROL': {'Other': 0.10, word_a: 0.15, word_b: 0.75, 'Total': 1}
    },
    'MODEL': {
        'A': {'Other': 0.00, word_a: 0.79, word_b: 0.21, 'Total': 1},
        'B': {'Other': 0.00, word_a: 0.13, word_b: 0.87, 'Total': 1},
        'CONTROL': {'Other': 0.00, word_a: 0.44, word_b: 0.56, 'Total': 1}
    }
}
# data = { # _AGE
#     'HUMANS': {
#         'A': {'Other': 0.35, word_a: 0.59, word_b: 0.06},
#         'B': {'Other': 0.57, word_a: 0.12, word_b: 0.31},
#         'CONTROL': {'Other': 0.78, word_a: 0.13, word_b: 0.09}
#     },
#     'MODEL': {
#         'A': {'Other': 0.33, word_a: 0.59, word_b: 0.08},
#         'B': {'Other': 0.37, word_a: 0.11, word_b: 0.52},
#         'CONTROL': {'Other': 0.70, word_a: 0.16, word_b: 0.14}
#     }
# }
# data = { # _OOL
#     'HUMANS': {
#         'A': {'Other': 0.48, word_a: 0.45, word_b: 0.07, 'Total': 1},
#         'B': {'Other': 0.28, word_a: 0.02, word_b: 0.70, 'Total': 1},
#         'CONTROL': {'Other': 0.78, word_a: 0.10, word_b: 0.12, 'Total': 1}
#     },
#     'MODEL': {
#         'A': {'Other': 0.39, word_a: 0.51, word_b: 0.10, 'Total': 1},
#         'B': {'Other': 0.36, word_a: 0.06, word_b: 0.58, 'Total': 1},
#         'CONTROL': {'Other': 0.69, word_a: 0.13, word_b: 0.18, 'Total': 1}
#     }
# }
# data = { # _REE
#     'HUMANS': {
#         'A': {'Other': 0.00, word_a: 0.87, word_b: 0.13, 'Total': 1},
#         'B': {'Other': 0.01, word_a: 0.15, word_b: 0.84, 'Total': 1},
#         'CONTROL': {'Other': 0.00, word_a: 0.46, word_b: 0.54, 'Total': 1}
#     },
#     'MODEL': {
#         'A': {'Other': 0.06, word_a: 0.81, word_b: 0.13, 'Total': 1},
#         'B': {'Other': 0.06, word_a: 0.13, word_b: 0.81, 'Total': 1},
#         'CONTROL': {'Other': 0.15, word_a: 0.42, word_b: 0.43, 'Total': 1}
#     }
# }

conditions = ['A', 'B', 'CONTROL']

response_types = [word_a, word_b, 'Other']
group_labels = ['HUMANS', 'MODEL']
colors = ['salmon', 'royalblue', 'lightgray']  # Match new stacking order

# Prep data for plotting
human_data = np.array([[data['HUMANS'][cond][resp] for resp in response_types] for cond in conditions])
model_data = np.array([[data['MODEL'][cond][resp] for resp in response_types] for cond in conditions])

bar_width = 0.35
r1 = np.arange(len(conditions))
r2 = [x + bar_width for x in r1]

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

# Labels and title
ax.set_xlabel('Group', fontsize=20, fontweight='bold')
ax.set_ylabel('Rate of Selection', fontsize=25, fontweight='bold')
ax.set_title('Humans Vs Model: ' + fragment, fontsize=25, fontweight='bold')
ax.set_xticks([r + bar_width/2 for r in range(len(conditions))])
ax.set_xticklabels(conditions, fontsize=16, fontweight='bold')

# Legend
ax.legend(title='Response Type')

# Vertical group labels
for label, x_locs, data_array in zip(group_labels, [r1, r2], [human_data, model_data]):
    for x, heights in zip(x_locs, data_array):
        total_height = sum(heights)
        ax.text(x, total_height / 2, label, ha='center', va='center', rotation='vertical', fontweight='bold', color='black', fontsize=18)

plt.tight_layout()
plt.show()
