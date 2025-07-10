import pandas as pd
import matplotlib.pyplot as plt

# 读取 data1.csv 数据
data = pd.read_csv('/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0423interference_11_1.csv')

# 检查数据列是否包含 'x' 和 'y' 列（假设磁铁位置的坐标存储在这些列）
if 'x' in data.columns and 'y' in data.columns:
    # 提取 x 和 y 坐标
    x = data['x']
    y = data['y']
    
    # 绘制磁铁在 XY 平面的轨迹图
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, color='blue', linewidth=2, label='Trajectory')  # 绘制轨迹线
    plt.scatter(x, y, color='red', s=10, alpha=0.5, label='Points')  # 用红点标记点
    
    # 添加标题和标签，并修改字体大小为 20
    plt.title('Magnet Trajectory in XY Plane', fontsize=20)
    plt.xlabel('X Position (cm)', fontsize=20)
    plt.ylabel('Y Position (cm)', fontsize=20)
    
    # 设置坐标轴刻度字体大小
    plt.tick_params(axis='both', labelsize=20)
    
    # 显示图例并修改字体大小为 20
    plt.legend(fontsize=20)
    
    # 显示网格和保持轴比例相等
    plt.grid(True)  # 添加网格线
    plt.axis('equal')  # 设置轴比例相等
    plt.show()
else:
    print("数据中不包含 'x' 和 'y' 列，请检查 CSV 文件格式。")
