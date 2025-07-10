import matplotlib.pyplot as plt
import numpy as np

# Define the categories
# labels_dynamic = [
#     "No Interference\nw/o",
#     "Small magnet (8*2)\nw/o",
#     "Small magnet (8*2)\nw/",
#     "New magnet (8*4)\nw/o",
#     "New magnet (8*4)\nw/"
# ]
# labels_dynamic = [
#     "No Interference\nw/o",
#     "Headphone Case\nw/o",
#     "Headphone Case\nw/",
#     "Reverse magnet\nw/o",
#     "Reverse magnet\nw/"
# ]
# labels_dynamic = [
#     "Magnet\nw/",
#     "Reverse magnet\nw/"
# ]
labels_dynamic = [
    # "Magnet\nw/",
    "Unet",
    "Ours"
]
# Tracking accuracy (can be updated to real values)
# accuracy_dynamic = [0.22, 2.25, 0.20, 3.28, 0.20]
accuracy_dynamic = [0.22, 1.25, 0.28, 3.33, 2.79]

accuracy_dynamic = [0.22, 1.25, 0.28, 3.33, 0.35]
accuracy_dynamic = [0.34, 0.34]
accuracy_dynamic = [0.22,0.21]
x_dyn = np.arange(len(labels_dynamic))  # label locations

# 设置颜色：w/o 为橙色，w/ 为绿色
colors = ['darkorange' if 'w/o' in label else '#1f77b4' for label in labels_dynamic]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(x_dyn, accuracy_dynamic, color=colors)

# 设置坐标轴标签、字体和网格
ax.set_ylabel('Tracking Accuracy (cm)', fontsize=20)
ax.set_xticks(x_dyn)
ax.set_xticklabels(labels_dynamic, fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# 在每个柱状图顶部标注数值
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

# 图例
handles = [
    plt.Rectangle((0, 0), 1, 1, color='darkorange', label='w/o Reconstruction'),
    plt.Rectangle((0, 0), 1, 1, color='#1f77b4', label='w/ Reconstruction')
]
ax.legend(handles=handles, fontsize=15)

plt.tight_layout()
plt.show()
