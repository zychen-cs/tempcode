import pandas as pd
import matplotlib
# matplotlib.use('Agg')  # 使用Agg后端

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as mticker
# # 读取测试数据
# test_df = pd.read_csv('/home/czy/桌面/magx-main1/Codes/optimization/alltest_LM_filtered.csv')
test_df = pd.read_csv('/home/czy/桌面/magx-main1/Codes/optimization/alltest.csv')
y_pred = pd.read_csv('/home/czy/桌面/magx-main1/Codes/optimization/ours_LMDiff_noise2_firstx.csv')

test_df = test_df[(test_df['theta'] == 0.0) & (test_df['phi'] == 0.0)]

# test_df = pd.read_csv('/home/czy/桌面/magx-main1/Codes/optimization/alltest_filtered.csv')
# y_pred = pd.read_csv('/home/czy/桌面/magx-main1/Codes/optimization/predictions.csv')
# # 筛选出 theta=0.0 且 phi=0.0 的数据
# filtered_df = test_df[(test_df['theta'] == 0.0) & (test_df['phi'] == 0.0)]

# # # 将筛选后的数据保存到新的CSV文件中
# filtered_df.to_csv('/home/czy/桌面/magx-main1/Codes/optimization/alltest_LM_filtered.csv', index=False)


# Ground truth 和预测数据


ground_truth = test_df[['magnet_x', 'magnet_y', 'magnet_z']].values
predicted = y_pred[['x', 'y', 'z']].values

# 计算 position error（欧式距离）
position_error = np.linalg.norm(ground_truth - predicted, axis=1)
mean_pos_error = np.mean(position_error)
print(mean_pos_error)



# ground truth θ/phi
theta_gt = test_df[['theta']].values
phi_gt = test_df[['phi']].values

# predicted θ/phi
theta_pred = y_pred[['theta']].values
phi_pred = y_pred[['phy']].values


# 球坐标转单位向量（x, y, z）
def spherical_to_unit_vec(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)  # shape: (N, 3)

vec_gt = spherical_to_unit_vec(theta_gt, phi_gt)
vec_pred = spherical_to_unit_vec(theta_pred, phi_pred)

# 计算单位向量夹角（单位为弧度）
dot_product = np.sum(vec_gt * vec_pred, axis=1)  # shape: (N,)
dot_product = np.clip(dot_product, -1.0, 1.0)  # 防止数值误差超出 arccos 定义域
angle_diff_rad = np.arccos(dot_product)  # 单位：弧度
angle_diff_deg = np.degrees(angle_diff_rad)  # 转为角度

mean_angle_error = np.mean(angle_diff_deg)
print(f"Mean Orientation Error (Angular Difference in Degrees): {mean_angle_error:.16f}")



test_df['position_error'] = position_error



# 计算样本的欧式距离
euclidean_distance = np.linalg.norm(test_df[['magnet_x', 'magnet_y', 'magnet_z']], axis=1)

# 定义距离区间
# distance_bins = [5, 7, 9, 11,13]
distance_bins = [5,6,7,8,9,10]
position_errors_by_distance = [[] for _ in distance_bins]  # 初始化双重列表

# 遍历所有样本，将 position error 分类到对应的距离区间
# for i, dist in enumerate(euclidean_distance):
#     error = position_error[i]
#     for j, bin_edge in enumerate(distance_bins):
#         if (dist <= bin_edge+0.1) & (dist >= bin_edge-0.1) :
#             position_errors_by_distance[j].append(error)
#             break  # 确保一个样本只进入一个区间
# print(np.mean(position_errors_by_distance[1]))
# print(np.mean(position_errors_by_distance[2]))
# print(np.mean(position_errors_by_distance[5]))
for i, dist in enumerate(euclidean_distance):
    error = position_error[i]
    for j, bin_edge in enumerate(distance_bins):
        if dist <= bin_edge:
            position_errors_by_distance[j].append(error)
            break  # 确保一个样本只进入一个区间

# 计算每个距离区间的误差均值
average_position_errors = [np.mean(errors) if errors else 0 for errors in position_errors_by_distance]
# data = {
#     'Distance (cm)': distance_bins,
#     'Average Position Error (cm)': average_position_errors
# }

# # 将字典转换为 DataFrame
# df = pd.DataFrame(data)

# 保存到 CSV 文件
# df.to_csv('average_position_errors_LM.csv', index=False)


# # 绘制柱状图
# x = np.arange(len(distance_bins))
# fig, ax = plt.subplots(figsize=(8, 6))

# # ax.bar(x, average_position_errors, color='#5D3A9B', alpha=0.7, label='Position Error')
# ax.bar(x, average_position_errors, color='#E69F00', alpha=0.7, label='Position Error')

# ax.set_xticks(x)
# # 设置坐标轴字体大小
# ax.set_xticks(x)
# ax.set_xticklabels([f'{d}' for d in distance_bins], fontsize=25)
# ax.set_yticklabels(ax.get_yticks(), fontsize=25)
# # 设置轴标签
# ax.set_xlabel('Distance (cm)', fontsize=25, labelpad=10)
# ax.set_ylabel('Error (cm)', fontsize=25, labelpad=10)
# ax = plt.gca()
# ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))  # 保留 1 位小数

# # ax.set_ylabel('Error (cm)')
# # ax.set_xlabel('Distance (cm)')
# # ax.legend(fontsize=20)

# plt.tight_layout()
# plt.savefig("errors_by_euclidean_distance_mlp.png")
# plt.show()
x = np.arange(len(distance_bins))
fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(x, average_position_errors, color='#5D3A9B', alpha=0.7, label='Position Error')

ax.set_xticks(x)
ax.set_xticklabels([f'{d}' for d in distance_bins], fontsize=25)
ax.set_xlabel('Distance (cm)', fontsize=25, labelpad=10)

# 设置 y 轴刻度
y_ticks = np.arange(0, max(average_position_errors) + 0.05, 0.1)  # 从 0 到最大值，步长为 1
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=25)

ax.set_ylabel('Error (cm)', fontsize=25, labelpad=10)
ax.legend(fontsize=20)

plt.tight_layout()
plt.grid(axis="y")
# plt.savefig("errors_by_euclidean_distance_mlp_smartphone.pdf")
# plt.savefig('Figure4(b).jpg', dpi=300)  # 保存插值结果图像为 PDF 格式
plt.show()