import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
mlp = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_mlp.csv')
# mlp1 = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_mlp_withoutinter.csv')
CNN = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_CNN.csv')
LM = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_LM.csv')

# data1 = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_mlp_dataset1.csv')
# data2 = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_mlp_dataset2.csv')
# data3 = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_mlp_dataset3.csv')
# data4 = pd.read_csv('/home/czy/桌面/magx-main1/average_position_errors_mlp_dataset4.csv')
# 提取数据
distance = mlp["Distance (cm)"]
# data1 = data1["Average Position Error (cm)"]
# data2 = data2["Average Position Error (cm)"]
# data3 = data3["Average Position Error (cm)"]
# data4 = data4["Average Position Error (cm)"]

mlpdata = mlp["Average Position Error (cm)"]
CNNdata = CNN["Average Position Error (cm)"]
LMdata = LM["Average Position Error (cm)"]
print(mlpdata)
print(CNNdata)
print(LMdata)
# mlpdata1 = mlp1["Average Position Error (cm)"]
# 确保所有的distance bins一致
distance_bins = distance  # 使用mlp的distance作为标准

# 创建图形
x = np.arange(len(distance_bins))
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制MLP, CNN, LM误差条形图
# ax.bar(x - 0.2, data1, width=0.2, color='#5D3A9B', alpha=0.7, label='5mm')  # MLP 使用 #5D3A9B
# ax.bar(x, data2, width=0.2, color='#E69F00', alpha=0.7, label='1cm')  # CNN 使用 #E69F00
# ax.bar(x + 0.2, data4, width=0.2, color='#56B4E9', alpha=0.7, label='1.5cm')  # LM 使用 #56B4E9（学术配色）
# ax.bar(x + 0.3, data3, width=0.2, color='#56B4E9', alpha=0.7, label='2cm')  # LM 使用 #56B4E9（学术配色）

# ax.bar(x - 0.1, mlpdata, width=0.2, color='#5D3A9B', alpha=0.7, label='w/')  # MLP 使用 #5D3A9B
# ax.bar(x+0.1, mlpdata1, width=0.2, color='#E69F00', alpha=0.7, label='w/o')  # CNN 使用 #E69F00

ax.bar(x - 0.2, mlpdata, width=0.2, color='#5D3A9B', alpha=0.7, label='MLP')  # MLP 使用 #5D3A9B
ax.bar(x, CNNdata, width=0.2, color='#E69F00', alpha=0.7, label='CNN')  # CNN 使用 #E69F00
ax.bar(x + 0.2, LMdata, width=0.2, color='#56B4E9', alpha=0.7, label='LM')  # LM 使用 #56B4E9（学术配色）

# 设置 x 轴标签和刻度
ax.set_xticks(x)
ax.set_xticklabels([f'{d}' for d in distance_bins], fontsize=25)
ax.set_xlabel('Distance (cm)', fontsize=25, labelpad=10)

# 设置 y 轴刻度
y_ticks = np.arange(0, max(max(mlpdata), max(CNNdata),max(LMdata)) + 0.05, 0.1)  # 从 0 到最大值，步长为 0.1
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=25)

# 设置 y 轴标签
ax.set_ylabel('Error (cm)', fontsize=25, labelpad=10)

# 添加图例
ax.legend(loc="upper center", fontsize=18, bbox_to_anchor=(0.5, 1.15), ncol=4)

# ax.legend(loc="upper right",fontsize=18)
plt.grid(axis="y")
# 使图形布局紧凑
plt.tight_layout()
# plt.savefig("Figure15.jpg",dpi=300)
# 显示图形
plt.show()
