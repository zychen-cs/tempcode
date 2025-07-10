import pandas as pd
import matplotlib.pyplot as plt

# 读取 data1.csv 数据
data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0216MLP4_1.csv')

# 真实 M 形状的点
points = [(-2.5, -4.5), (-1.5, -6.5), (0, -5), (1.5, -6.5), (2.5, -4.5)]
x_m, y_m = zip(*points)

# 创建图形
plt.figure(figsize=(10, 10))

# 绘制真实 M 形状（红色曲线 + 圆点）
plt.plot(x_m, y_m, marker='o', linestyle='-', color='red', linewidth=2, label='Ground Truth')

# 检查数据列是否包含 'x' 和 'y' 列
if 'x' in data.columns and 'y' in data.columns:
    # 提取 x 和 y 坐标
    x_tracked = data['x']
    y_tracked = data['y']
    
    # 绘制跟踪的轨迹（蓝色曲线 + 红色散点）
    plt.plot(x_tracked, y_tracked, color='blue', linewidth=2, label='Tracked Trajectory')
    # plt.scatter(x_tracked, y_tracked, color='red', s=10, alpha=0.5, label='Tracked Points')

# 添加标题和标签
# plt.title('Magnet Trajectory vs. True M Shape', fontsize=20)
plt.xlabel('X Position (cm)', fontsize=22)
plt.ylabel('Y Position (cm)', fontsize=22)

# 设置坐标轴刻度字体大小
plt.tick_params(axis='both', labelsize=22)

# 显示图例
plt.legend(fontsize=22)

# 显示网格和保持轴比例相等
plt.grid(True)
plt.axis('equal')
plt.savefig("airwriting_M.pdf")
# 显示图形
plt.show()
