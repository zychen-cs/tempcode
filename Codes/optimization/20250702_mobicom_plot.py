import matplotlib.pyplot as plt
import numpy as np

# 数据（由远到近）
distances = ['Far', 'Mid', 'Near']
axes = ['X', 'Y', 'Z']

# 每个轴上三个距离点的误差值
position_errors = {
    'X': [0.59, 0.69, 0.81],
    'Y': [0.69, 0.75, 0.81],
    'Z': [0.69, 0.87, 0.71],
}

orientation_errors = {
    'X': [1.65, 1.80, 2.02],
    'Y': [1.51, 1.64, 2.02],
    'Z': [1.80, 1.55, 1.74],
}

# 将数据转换为矩阵（行=轴，列=距离）
pos_data = np.array([position_errors[axis] for axis in axes])
ori_data = np.array([orientation_errors[axis] for axis in axes])

# 画图参数
bar_width = 0.25
x = np.arange(len(distances))

# ---------- 图 1：Position Error ----------
fig, ax = plt.subplots(figsize=(8, 5))
for i in range(len(axes)):
    ax.bar(x + i * bar_width, pos_data[i], width=bar_width, label=f'{axes[i]}-axis')

# 坐标设置
ax.set_xlabel('Distance (Far → Near)', fontsize=12)
ax.set_ylabel('Mean Position Error (Euclidean Distance)', fontsize=12)
ax.set_title('Position Error Across Axes and Distances', fontsize=14)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(distances)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)

# 显示数值标签
for i in range(len(axes)):
    for j in range(len(distances)):
        ax.text(x[j] + i * bar_width, pos_data[i][j] + 0.01,
                f'{pos_data[i][j]:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ---------- 图 2：Orientation Error ----------
fig, ax = plt.subplots(figsize=(8, 5))
for i in range(len(axes)):
    ax.bar(x + i * bar_width, ori_data[i], width=bar_width, label=f'{axes[i]}-axis')

# 坐标设置
ax.set_xlabel('Distance (Far → Near)', fontsize=12)
ax.set_ylabel('Mean Orientation Error (Degrees)', fontsize=12)
ax.set_title('Orientation Error Across Axes and Distances', fontsize=14)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(distances)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)

# 显示数值标签
for i in range(len(axes)):
    for j in range(len(distances)):
        ax.text(x[j] + i * bar_width, ori_data[i][j] + 0.05,
                f'{ori_data[i][j]:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
