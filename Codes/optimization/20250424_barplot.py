import matplotlib.pyplot as plt
import numpy as np

# Define the categories
# labels = [
#     "Small magnet (8*2)\nNo Reconstruction",
#     "Small magnet (8*2)\nWith Reconstruction",
#     "New magnet (8*4)\nNo Reconstruction",
#     "New magnet (8*4)\nWith Reconstruction"
# ]
labels = [
    "No Interference\nNo Reconstruction",
    "No Interference\nWith Reconstruction",
    
]
# Tracking accuracy values (cm), split into two metrics for each condition
# accuracy_1 = [1.06, 0.25, 2.13, 0.31]  # First metric
# accuracy_2 = [3.38, 0.45, 3.92, 0.40]  # Second metric
accuracy_1 = [1.08, 0.11, 1.68, 0.16]  # First metric
accuracy_2 = [3.16, 0.20, 3.65, 0.60]  # Second metric

accuracy_1 = [0.12, 0.28]  # First metric
accuracy_2 = [0.25, 0.60]  # Second metric
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, accuracy_1, width, label='Pos 1')
rects2 = ax.bar(x + width/2, accuracy_2, width, label='Pos 2')

# Add text for labels, title and axes ticks
ax.set_ylabel('Tracking Accuracy (cm)', fontsize=20)
# ax.set_title('Tracking Accuracy by Interference and Reconstruction Module', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.legend(fontsize=15)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()
