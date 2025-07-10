import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0216MLP4_1.csv')

# 真实 M 形状的轨迹点
points = [(-2.5, -4.5), (-1.5, -6.5), (0, -5), (1.5, -6.5), (2.5, -4.5)]

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

# 计算误差
errors = []
for _, row in data.iterrows():
    x_tracked, y_tracked = row['x'], row['y']
    
    # 计算当前点到所有目标轨迹点的距离，取最小值
    distances = np.sqrt((dense_trajectory[:, 0] - x_tracked) ** 2 + (dense_trajectory[:, 1] - y_tracked) ** 2)
    min_distance = np.min(distances)
    
    errors.append(min_distance)

# 统计误差
mean_error = np.mean(errors)
std_error = np.std(errors)
print(f"平均误差: {mean_error:.4f} cm")
print(f"误差标准差: {std_error:.4f} cm")

# 绘制轨迹
plt.figure(figsize=(10, 10))
plt.plot(dense_trajectory[:, 0], dense_trajectory[:, 1], marker='.', linestyle='-', color='red', markersize=2, label='Interpolated Ground Truth')
plt.plot(data['x'], data['y'], color='blue', linewidth=2, label='Tracked Trajectory')

plt.xlabel('X Position (cm)', fontsize=22)
plt.ylabel('Y Position (cm)', fontsize=22)
plt.tick_params(axis='both', labelsize=22)
plt.legend(fontsize=22)
plt.grid(True)
plt.axis('equal')
plt.show()

# 误差分布
plt.hist(errors, bins=20, edgecolor='black')
plt.xlabel('误差 (cm)', fontsize=22)
plt.ylabel('频率', fontsize=22)
plt.tick_params(axis='both', labelsize=22)
plt.title('误差分布', fontsize=22)
plt.show()
