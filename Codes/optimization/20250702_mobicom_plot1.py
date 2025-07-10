import matplotlib.pyplot as plt
import numpy as np

# 数据（由远到近）
axes = ['X', 'Y', 'Z']
distances = ['Far', 'Mid', 'Near']

# 每个轴上三个距离点的误差值（从远到近）
#8*2 w/
position_errors = {
    'X': [0.59, 0.69, 0.81],
    'Y': [0.69, 0.75, 0.69],
    'Z': [0.71, 0.87, 0.69],
}

orientation_errors = {
    'X': [1.65, 1.80, 2.02],
    'Y': [1.51, 1.64, 1.80],
    'Z': [1.74, 1.55, 1.80],
}

#8*2 w/o
position_errors = {
    'X': [0.59,1.58,42.43 ],
    'Y': [0.82,1.35,1.58],
    'Z': [1.34,1.51,1.58],
}

orientation_errors = {
    'X': [1.54,5.05,18.02],
    'Y': [2.44,4.43,5.05],
    'Z': [11.40,7.97,5.05],
}

# #8*8 w/
position_errors = {
    'X': [0.81,0.72,0.90],
    'Y': [0.96,0.70,0.72],
    'Z': [0.64,0.75,0.72],
}

orientation_errors = {
    'X': [1.61,1.97,1.96],
    'Y': [1.87,1.80,1.97],
    'Z': [1.86,1.84,1.97],
}

# #8*8 w/o
# position_errors = {
#     'X': [2.35,71.16,315.97],
#     'Y': [10.46,61.95,71.16],
#     'Z': [69.14,201.61,71.16],
# }

# orientation_errors = {
#     'X': [9.23,19.82,38.27],
#     'Y': [24.83,25.07,19.82],
#     'Z': [26.16,28.49,19.82],
# }

# #L-type w/
# position_errors = {
#     'X': [0.59,0.62,0.67],
#     'Y': [],
#     'Z': [],
# }

# orientation_errors = {
#     'X': [1.77,1.78,1.89],
#     'Y': [],
#     'Z': [],
# }

# 将数据转换为矩阵（行=距离，列=轴）
pos_data = np.array([[position_errors[axis][i] for axis in axes] for i in range(3)])
ori_data = np.array([[orientation_errors[axis][i] for axis in axes] for i in range(3)])

# 绘图参数
bar_width = 0.25
x = np.arange(len(axes))  # X, Y, Z
# colors = ['#5D3A9B', '#E69F00', '#56B4E9']  # 代表不同距离的颜色
colors = ['#5D3A9B', '#E69F00', '#009E73']  # 代表不同距离的颜色
# ---------- 图 1：Position Error ----------
fig, ax = plt.subplots(figsize=(8, 5))
for i in range(3):  # 遍历距离 Far, Mid, Near
    ax.bar(x + i * bar_width, pos_data[i], width=bar_width, label=distances[i], color=colors[i],alpha=0.7)

# 设置图例和坐标轴
# ax.set_xlabel('Axis', fontsize=22)
ax.set_ylabel('Position Error (cm)', fontsize=22)
ax.tick_params(axis='x', labelsize=22)   # X-axis tick font size
ax.tick_params(axis='y', labelsize=22)   # Y-axis tick font size
# ax.set_title('Position Error by Axis and Distance', fontsize=14)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(axes)
ax.legend(title='Distance',fontsize=16,title_fontsize=18,loc="upper right",framealpha=0.5)
# ax.legend(title='Distance',fontsize=13,title_fontsize=15,bbox_to_anchor=(0.35, 1.3), ncol=4)
ax.grid(axis='y', linestyle='--', alpha=0.4)

# 添加误差标签
for i in range(3):
    for j in range(3):
        ax.text(x[j] + i * bar_width, pos_data[i][j] + 0.01,
                f'{pos_data[i][j]:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("5_3_2_pos82_wo.pdf")
plt.show()

# ---------- 图 2：Orientation Error ----------
fig, ax = plt.subplots(figsize=(8, 5))
for i in range(3):  # 遍历距离
    ax.bar(x + i * bar_width, ori_data[i], width=bar_width, label=distances[i], color=colors[i],alpha=0.7)

# ax.set_xlabel('Axis', fontsize=22)
ax.set_ylabel('Orientation Error (°)', fontsize=22)
ax.tick_params(axis='x', labelsize=22)   # X-axis tick font size
ax.tick_params(axis='y', labelsize=22)   # Y-axis tick font size
# ax.set_title('Orientation Error by Axis and Distance', fontsize=20)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(axes)
# ax.legend(title='Distance')
ax.legend(title='Distance',fontsize=16,title_fontsize=18,loc="upper right",framealpha=0.5)
# ax.legend(title='Distance',fontsize=13,title_fontsize=15,bbox_to_anchor=(0.3, 1), ncol=4)
ax.grid(axis='y', linestyle='--', alpha=0.4)

# 添加误差标签
for i in range(3):
    for j in range(3):
        ax.text(x[j] + i * bar_width, ori_data[i][j] + 0.05,
                f'{ori_data[i][j]:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("5_3_2_ori82_wo.pdf")
plt.show()
