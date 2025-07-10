import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0216LM_I_1.csv')
# data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0216MLP_Znewmag1_1.csv')
data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0216ring_O_1.csv')
# data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0423interference_9_1.csv')

error_raw=[0.0823,0.1570,0.0740,0.1282,0.1056,0.1213,0.1039,0.1483,0.0397,0.1841,0.0959,0.0246,0.1265,0.0715,0.1316,0.0574,0.1933,0.0877,0.1401,0.0731,0.1055,0.0526,0.1035,0.1331,0.0629,0.0950 ]
error_smooth=[0.0782, 0.1617,0.0705,0.1275,0.1024,0.1163,0.0957,0.1506,0.0390,0.1804,0.0936,0.0248,0.1191,0.0653,0.1312,0.0566,0.1892,0.0846,0.1349,0.0710,0.1018,0.0486,0.1008,0.1258,0.0615,0.0882 ]
# 真实 M 形状的轨迹点
# 定义字母与坐标的映射
letter_points = {
    # "A": [(-1.5, -4.5), (0, -6.5), (1.5, -4.5), (0.75, -5.5), (-0.75, -5.5)],
    "A": [(2, -4.5), (0, -6.5), (-2, -4.5), (-1, -5.5), (1, -5.5)],
    "B": [(0, -4.5), (0, -6.5), (-1, -6.5), (-1, -5.5), (0, -5.5), (-1, -5.5), (-1, -4.5),(0, -4.5)],
    "C": [(-1, -6.5), (0, -6.5), (1, -6), (1, -5), (0, -4.5), (-1, -4.5)],
    "D": [(0, -4.5), (0, -6.5), (-1, -6.5), (-2, -6), (-2, -5), (-1, -4.5), (0, -4.5)],
    "E": [(-1, -6.5), (0, -6.5), (0, -5.5), (-1, -5.5), (0, -5.5), (0, -4.5), (-1, -4.5)],
    "F": [(-1, -6.5), (0, -6.5), (0, -5.5), (-1, -5.5), (0, -5.5), (0, -4.5)],
    "G": [(-1, -6.5), (0, -6.5), (1, -6), (1, -5), (0, -4.5), (-1, -4.5), (-1, -5.5), (0, -5.5)],
    "H": [(0, -6.5), (0, -4.5), (0, -5.5), (-1, -5.5), (-1, -6.5), (-1, -4.5)],
    "I": [(0, -6.5), (0, -4.5)],
    "J": [(-1, -6.5), (-1, -4.5), (0, -4.5), (1, -5)],
    "K": [(-1, -6.5), (0, -5.5), (0, -6.5), (0, -4.5), (0, -5.5), (-1, -4.5)],
    "L": [(0, -6.5), (0, -4.5), (-1, -4.5)],
    # "M": [(-2.5, -4.5), (-1.5, -6.5), (0, -5), (1.5, -6.5), (2.5, -4.5)],
    "M": [(2, -4.5), (1, -5.5), (0, -4.5), (-1, -5.5), (-2, -4.5)],
    "N": [(0, -4.5), (0, -6.5), (-1, -4.5), (-1, -6.5)],
    "O": [(-1, -6.5), (0, -6.5), (0, -4.5), (-1, -4.5),(-1, -6.5)],
    "P": [(0, -5.5), (-1, -5.5), (-1, -6.5), (0, -6.5), (0, -4.5)],
    "Q": [(-1, -5.5), (-1, -6.5), (0, -6.5), (0, -5.5), (-1, -5.5), (-2, -5)],
    "R": [(0, -5.5), (-1, -5.5), (-1, -6.5), (0, -6.5), (0, -4.5), (0, -5.5), (-1, -4.5)],
    "S": [(-1, -6.5), (0, -6.5), (0, -5.5), (-1, -5.5), (-1, -4.5), (0, -4.5)],
    "T": [(1, -6.5), (-1, -6.5), (0, -6.5), (0, -4.5)],
    "U": [(1, -6.5), (1, -4.5), (-1, -4.5), (-1, -6.5)],
    "V": [(1, -5.5), (0, -4.5), (-1, -5.5)],
    "W": [(2, -5.5), (1, -4.5), (0, -5.5), (-1, -4.5), (-2, -5.5)],
    "X": [(1, -6.5), (-1, -4.5), (0, -5.5), (-1, -6.5), (1, -4.5)],
    "Y": [(1, -6.5), (0, -5.5), (-1, -6.5), (0, -5.5), (0, -4.5)],
    "Z": [(1, -6.5), (-1, -6.5), (1, -4.5), (-1, -4.5)],
    "T1": [(-1, -6.5), (1, -6.5), (1, -4.5), (-1, -4.5),(-1, -6.5)]
}

# 获取所需字母的坐标
def get_letter_points(letter):
    return letter_points.get(letter.upper(), None)
error_new=[0.1110,0.0719,0.0597,0.0691,0.0959,0.0995,0.1126,0.0792,0.0458,0.2046,0.0541,0.0693,0.1787,0.0678,0.1043,0.0932,0.2387,0.0826,0.0937,0.0373,
           0.1068,0.1129,0.1210,0.1140,0.0767,0.0584 ]
# 示例：获取字母 A 的点
selected_letter = "O"
points = get_letter_points(selected_letter)

# 生成高密度的目标轨迹点
dense_trajectory = []
num_samples = 100  # 每段插值为100个点
for i in range(len(points) - 1):
    x1, y1 = points[i]
    x2, y2 = points[i + 1]
    x_interp = np.linspace(x1, x2, num_samples)
    y_interp = np.linspace(y1, y2, num_samples)
    dense_trajectory.extend(zip(x_interp, y_interp))

dense_trajectory = np.array(dense_trajectory)  # (N,2) 形状的数组

# 实时指数移动平均 (EMA) 平滑
alpha = 0.2  # 平滑系数，取值越小越平滑
data['x_ema'] = data['x'].copy()
data['y_ema'] = data['y'].copy()

for i in range(1, len(data)):
    data.at[i, 'x_ema'] = alpha * data.at[i, 'x'] + (1 - alpha) * data.at[i-1, 'x_ema']
    data.at[i, 'y_ema'] = alpha * data.at[i, 'y'] + (1 - alpha) * data.at[i-1, 'y_ema']

# 计算误差（原始数据）
errors_raw = []
errors_ema = []
for _, row in data.iterrows():
    x_raw, y_raw = row['x'], row['y']
    x_smooth, y_smooth = row['x_ema'], row['y_ema']

    distances_raw = np.sqrt((dense_trajectory[:, 0] - x_raw) ** 2 + (dense_trajectory[:, 1] - y_raw) ** 2)
    distances_smooth = np.sqrt((dense_trajectory[:, 0] - x_smooth) ** 2 + (dense_trajectory[:, 1] - y_smooth) ** 2)

    errors_raw.append(np.min(distances_raw))
    errors_ema.append(np.min(distances_smooth))

# 统计误差
mean_error_raw = np.mean(errors_raw)
std_error_raw = np.std(errors_raw)
mean_error_ema = np.mean(errors_ema)
std_error_ema = np.std(errors_ema)

print(f"原始数据 - 平均误差: {mean_error_raw:.4f} cm, 误差标准差: {std_error_raw:.4f} cm")
print(f"实时平滑 - 平均误差: {mean_error_ema:.4f} cm, 误差标准差: {std_error_ema:.4f} cm")

# 绘制原始轨迹 vs 平滑轨迹
plt.figure(figsize=(10, 10))
plt.plot(dense_trajectory[:, 0], dense_trajectory[:, 1], marker='.', linestyle='-', color='red', markersize=2, label='Ground Truth')
# plt.plot(data['x'], data['y'], color='blue', linewidth=2, alpha=0.5, label='Tracked Trajectory (Raw)')
plt.plot(data['x_ema'], data['y_ema'], color='green', linewidth=2, label='Tracked Trajectory')




plt.xlabel('X Position (cm)', fontsize=22)
plt.ylabel('Y Position (cm)', fontsize=22)
plt.tick_params(axis='both', labelsize=22)
plt.legend(fontsize=22)
plt.grid(True)
plt.axis('equal')
# plt.title("轨迹对比：原始 vs. 实时平滑", fontsize=22)
plt.xlim(-10, 10)  # 扩大 x 轴范围
plt.ylim(-10, 0)  # 扩大 y 轴范围
# plt.tight_layout()
# plt.savefig("Figure10_c.jpg",dpi=300)
# plt.savefig("airwriting_Z_ring.pdf")
plt.show()

# 误差分布对比
# plt.figure(figsize=(12, 5))
# plt.hist(errors_raw, bins=20, alpha=0.5, label='Raw Errors', edgecolor='black')
# plt.hist(errors_ema, bins=20, alpha=0.5, label='EMA Smoothed Errors', edgecolor='black')
# plt.xlabel('误差 (cm)', fontsize=22)
# plt.ylabel('频率', fontsize=22)
# plt.tick_params(axis='both', labelsize=22)
# plt.title('误差分布对比', fontsize=22)
# plt.legend(fontsize=22)
# plt.show()
