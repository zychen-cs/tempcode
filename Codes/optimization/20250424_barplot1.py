import matplotlib.pyplot as plt
import numpy as np
# New dataset for the second table (Dynamic Target Magnet)

# Define the categories
labels_dynamic = [
    "No Interference\nw/o",
    "Small magnet (8*2)\nw/o",
    "Small magnet (8*2)\nw/",
    "New magnet (8*4)\nw/o",
    "New magnet (8*4)\nw/"
]

# Only one metric is given in this case
accuracy_dynamic = [0.22, 2.25, 0.20, 3.28, 0.20]
accuracy_dynamic = [0.22, 0.76, 0.27, 1.22, 0.24]
x_dyn = np.arange(len(labels_dynamic))  # the label locations

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(x_dyn, accuracy_dynamic, color='darkorange')

# Add text for labels, title and axes ticks
ax.set_ylabel('Tracking Accuracy (cm)', fontsize=20)
# ax.set_title('Tracking Accuracy (Dynamic Target Magnet)', fontsize=20)
ax.set_xticks(x_dyn)
ax.set_xticklabels(labels_dynamic, fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()
