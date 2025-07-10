import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
mlp = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_mlp.csv')
mlp1 = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_mlp_alldir.csv')
# CNN = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_CNN.csv')
# LM = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_LM.csv')


# data1 = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_mlp_dataset1.csv')
# data2 = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_mlp_dataset2.csv')
# data3 = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_mlp_dataset3.csv')
# data4 = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_mlp_dataset4.csv')
# 提取数据
distance = mlp["Distance (cm)"]
data1 = mlp["Average Position Error (cm)"]
data2 = mlp1["Average Position Error (cm)"]

# CNNdata = CNN["Average Position Error (cm)"]
# LMdata = LM["Average Position Error (cm)"]

# 确保所有的distance bins一致
distance_bins = distance  # 使用mlp的distance作为标准

# 创建图形
x = np.arange(len(distance_bins))
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制MLP, CNN, LM误差条形图


ax.bar(x - 0.1, data1, width=0.2, color='#5D3A9B', alpha=0.7, label='One Direction Data ')  # MLP 使用 #5D3A9B
ax.bar(x+0.1, data2, width=0.2, color='#E69F00', alpha=0.7, label='Five Directions Data')  # CNN 使用 #E69F00
# Single Magnet Orientation" vs. "Multiple Magnet Orientations


# 设置 x 轴标签和刻度
ax.set_xticks(x)
ax.set_xticklabels([f'{d}' for d in distance_bins], fontsize=25)
ax.set_xlabel('Distance (cm)', fontsize=25, labelpad=10)

# 设置 y 轴刻度
y_ticks = np.arange(0, max(max(data1), max(data2)) + 0.05, 0.1)  # 从 0 到最大值，步长为 0.1
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=25)

# 设置 y 轴标签
ax.set_ylabel('Error (cm)', fontsize=25, labelpad=10)

# 添加图例
# ax.legend(loc="upper right",fontsize=18)
ax.legend(loc="upper center", fontsize=16, bbox_to_anchor=(0.5, 1.15), ncol=4)
plt.grid(axis="y")
# 使图形布局紧凑
plt.tight_layout()
# plt.savefig("Figure12.jpg",dpi=300)
# 显示图形
plt.show()
