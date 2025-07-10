import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.interpolate import interp1d
data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0216LM2_1.csv')
data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0216LM_A_1.csv')
data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0216MLP_Znewenv2_1.csv')
data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0216CNN_Z_1.csv')
data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0216ring_E_1.csv')
data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0303test_A1_1.csv')
# data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0216MLP_A_1.csv')
# 确保数据列存在
if not {'x', 'y'}.issubset(data.columns):
    raise ValueError("CSV 文件必须包含 'x' 和 'y' 列")

x = data['x'].values
y = data['y'].values
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
    "Z": [(1, -6.5), (-1, -6.5), (1, -4.5), (-1, -4.5)]
}

# 获取所需字母的坐标
def get_letter_points(letter):
    return letter_points.get(letter.upper(), None)
error_new=[0.1110,0.0719,0.0597,0.0691,0.0959,0.0995]
error_MLP_newdir=[0.33,0.31,0.14,0.21,0.41,0.37,0.27,0.19,0.05,0.24,0.15,0.10,0.31,0.10,0.13,0.21,0.36,0.24,0.23,0.07,0.14,0.14,0.20,0.19,0.19,0.15]
error_MLP_newmag=[0.29,0.54,0.67,0.58,0.71,0.65,0.68,0.62,0.75,0.38,0.59,0.67,0.42,0.41,0.42,0.59,0.22,0.71,0.32,0.54,0.30,0.37,0.37,0.43,0.66,0.50]
#B-
error_MLP=[0.14,0.55,0.14,0.40,0.36,0.26,0.25,0.26,0.05,0.23,0.24,0.04,0.15,0.10,0.13,0.16,0.31,0.24,0.30,0.13,0.13,0.08,0.20,0.20,0.16,0.14]
error_MLP_newenv1=[0.17,0.31,0.19,0.21,0.29,0.32,0.27,0.23,0.19,0.24,0.29,0.16,0.20,0.21,0.18,0.29,0.24,0.25,0.18,0.21,0.20,0.21,0.12,0.21,0.26,0.15]
error_MLP_newenv2=[0.25,0.37,0.21,0.24,0.34,0.29,0.35,0.28,0.15,0.33,0.17,0.11,0.15,0.29,0.21,0.18,0.19,0.17,0.28,0.14,0.21,0.19,0.14,0.16,0.19,0.18]
error_CNN=[0.27,0.66,0.27,0.36,0.45,0.53,0.41,0.30,0.22,0.28,0.31,0.19,0.25,0.34,0.34,0.32,0.27,0.48,0.52,0.18,0.25,0.26,0.21,0.27,0.23,0.26]
#B-
error_LM=[0.11,0.21,0.44,0.24,0.66,0.11,0.23,1.32,0.18,0.10,0.70,0.30,0.50,0.73,0.98,0.47,0.10,0.76,0.51,0.25,0.70,0.43,0.20,0.50,0.64,0.23]
error_ring_MLP=[0.17,0.20,0.23,0.17,0.20,0.28,0.28,0.18,0.03,0.24,0.22,0.16,0.14,0.10,0.21,0.20,0.29,0.27,0.23,0.10,0.14,0.20,0.16,0.20,0.15,0.16]

zhy=[0.22,0.17,0.12,0.21,0.23]
cph=[0.29,0.20,0.14,0.26,0.15]
sww=[0.29,0.19,0.14,0.26,0.17]
wjk=[0.14,0.14,0.14,0.29,0.24]
hjy=[0.29,0.19,0.16,0.19,0.22]
wsy=[0.25,0.22,0.26,0.29,0.15]
cxm=[0.24,0.17,0.10,0.26,0.12]
czy=[0.14,0.15,0.10,0.31,0.14]

# 示例：获取字母 A 的点
selected_letter = "A"
points = get_letter_points(selected_letter)
dense_trajectory = []
num_samples = 100  # 每段插值为100个点
for i in range(len(points) - 1):
    x1, y1 = points[i]
    x2, y2 = points[i + 1]
    x_interp = np.linspace(x1, x2, num_samples)
    y_interp = np.linspace(y1, y2, num_samples)
    dense_trajectory.extend(zip(x_interp, y_interp))

dense_trajectory = np.array(dense_trajectory)  # (N,2) 形状的数组
alpha = 0.2
smoothed_x, smoothed_y = [x[0]], [y[0]]  # 初始点
for i in range(1, len(x)):
    smoothed_x.append(alpha * x[i] + (1 - alpha) * smoothed_x[-1])
    smoothed_y.append(alpha * y[i] + (1 - alpha) * smoothed_y[-1])

# 计算 DTW 距离
trajectory = list(zip(x, y))  # 传感器记录的轨迹
trajectory1 = list(zip(smoothed_x, smoothed_y))  # 传感器记录的轨迹
dtw_distance, _ = fastdtw(trajectory, dense_trajectory, dist=euclidean)
dtw_distance1, _ = fastdtw(trajectory1, dense_trajectory, dist=euclidean)
# dtw_distance, path = fastdtw(dense_trajectory, trajectory, dist=euclidean)
# print("匹配路径:", path)
# 计算 EMA 平滑轨迹
# alpha = 0.2
# smoothed_x, smoothed_y = [x[0]], [y[0]]  # 初始点
# for i in range(1, len(x)):
#     smoothed_x.append(alpha * x[i] + (1 - alpha) * smoothed_x[-1])
#     smoothed_y.append(alpha * y[i] + (1 - alpha) * smoothed_y[-1])

# 可视化轨迹对比
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b.-', label='原始轨迹')
plt.plot(dense_trajectory[:, 0], dense_trajectory[:, 1], 'r--', label='目标轨迹 (插值)')
plt.plot(smoothed_x, smoothed_y, 'g-', label='EMA 平滑轨迹')
# plt.legend()
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title(f'Error: {dtw_distance:.2f} cm')
print(dtw_distance/len(data))
print(dtw_distance1/len(data))
plt.grid()
plt.show()
