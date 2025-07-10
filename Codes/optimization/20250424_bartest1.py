import matplotlib.pyplot as plt
import numpy as np

# 类别标签
labels = [
    "No Interference\nNo Reconstruction",
    "No Interference\nWith Reconstruction",
]

# 准确率数据
accuracy_1 = [0.12, 0.28]  # Pos 1
accuracy_2 = [0.25, 0.60]  # Pos 2

x = np.arange(len(labels))
width = 0.32

fig, ax = plt.subplots(figsize=(12, 6))

# 设置颜色：根据是否含有 'With Reconstruction'
bar_colors = ['steelblue' if 'With Reconstruction' not in label else 'forestgreen' for label in labels]

# Pos 1 和 Pos 2 使用相同颜色序列
rects1 = ax.bar(x - width/2, accuracy_1, width, label='Pos 1', color=bar_colors)
rects2 = ax.bar(x + width/2, accuracy_2, width, label='Pos 2', color=bar_colors)

# 添加坐标轴标签和样式
ax.set_ylabel('Tracking Accuracy (cm)', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# 自定义图例（仅标识 Reconstruction 状态）
custom_legend = [
    plt.Rectangle((0, 0), 1, 1, color='steelblue', label='No Reconstruction'),
    plt.Rectangle((0, 0), 1, 1, color='forestgreen', label='With Reconstruction'),
]
ax.legend(handles=custom_legend, fontsize=15)

# 添加柱子顶部数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()
