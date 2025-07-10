import matplotlib
matplotlib.use('Agg')  # 使用Agg后端

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as mticker



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # 输入层到第一个隐藏层
        self.fc1 = nn.Linear(84, 256)  # 输入 21 维，扩大到 256
        self.bn1 = nn.BatchNorm1d(256)  # 批归一化
        
        # 第二个隐藏层
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        # 第三个隐藏层
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # 第四个隐藏层
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        # 第五个隐藏层
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        
        # Dropout层
        self.dropout = nn.Dropout(0.3)
        
        # 输出层
        self.fc6 = nn.Linear(32, 5)  # 假设有 5 个类别
        
        # 添加残差连接
        self.residual_fc1 = nn.Linear(256, 128)
        self.residual_fc2 = nn.Linear(128, 64)

    def forward(self, x):
        # 第一层：全连接 + 激活 + 批归一化
        x1 = F.gelu(self.bn1(self.fc1(x)))
        
        # 第二层：全连接 + 残差连接 + 激活 + 批归一化
        residual1 = self.residual_fc1(x1)
        x2 = F.gelu(self.bn2(self.fc2(x1)) + residual1)
        
        # 第三层：全连接 + 激活 + 批归一化
        x3 = F.gelu(self.bn3(self.fc3(x2)))
        
        # 第四层：全连接 + 残差连接 + 激活 + 批归一化
        residual2 = self.residual_fc2(x2)
        x4 = F.gelu(self.bn4(self.fc4(x3)) + residual2)
        
        # 第五层：全连接 + 激活 + 批归一化
        x5 = self.dropout(F.gelu(self.bn5(self.fc5(x4))))
        
        # 输出层
        x_out = self.fc6(x5)
        return x_out



# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)
# model.load_state_dict(torch.load('../magnetic/model/1122full_data_mlp_model_v1.pth'))
# model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0130_differential_mlp_model_v3.pth',map_location=torch.device('cpu')))
# model.load_state_dict(torch.load('/home/wangmingke/czy/magnetic/0220_differential_mlp_model_v1.pth'))
# model.load_state_dict(torch.load('/home/wangmingke/czy/magnetic/0301_differential_mlp_model_lesszdata_v4.pth'))
# model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0301_differential_mlp_model_alldir_v1.pth',map_location=torch.device('cpu')))
model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0613model_olddata.pth',map_location=torch.device('cpu')))


model.eval()

# 读取测试数据 
# test_df = pd.read_csv('/home/wangmingke/czy/magnetic/benchdataset/combined_differential_dataset_1cm.csv')

# 读取测试数据
test_df = pd.read_csv('/home/czy/桌面/magx-main1/normaldata/alltest.csv')
# test_df = pd.read_csv('/home/czy/桌面/magx-main1/thirdx_test.csv')
# 筛选出 theta=0.0 且 phi=0.0 的数据
test_df = test_df[(test_df['theta'] == 0.0) & (test_df['phi'] == 0.0)]
# test_df = test_df[(test_df['theta'] == 0.0) & (test_df['phi'] == 0.0)&
#     (test_df['magnet_z']>=0)]
# test_df = test_df[(test_df['theta'] == 0.0) & (test_df['phi'] == 0.0)&
#     (test_df['magnet_z']<0)]
# test_df = test_df[(test_df['theta'].isin([0.0, 0])) & (test_df['phi'].isin([0.0, 0]))]
print(len(test_df))

# test_df = pd.read_csv('/home/wangmingke/czy/magnetic/mappingdata/dataset_0.5cm_diff.csv')
# 假设数据中有 'y' 列，这里是目标列。去除 y = -9 和 y = -10 的数据
# test_df = test_df[(test_df['magnet_y'] != -9) & (test_df['magnet_y'] != -10)]

# test_df = pd.read_csv('/home/wangmingke/czy/magnetic/benchdataset/differential_dataset1.csv')

# X = df[[f'sensor_{i+1}_Bx' for i in range(8) if i != 6] +
#        [f'sensor_{i+1}_By' for i in range(8) if i != 6] +
#        [f'sensor_{i+1}_Bz' for i in range(8) if i != 6]].values


# 假设 df 是原始数据框，sensors 是传感器列表或序列
sensors = [f'sensor_{i+1}' for i in range(8)]  # 传感器名称列表

# 初始化一个空的列表来存储列名
columns = []

# 遍历所有传感器对
for i in range(len(sensors)):
    for j in range(i + 1, len(sensors)):
        # 生成每对传感器的 Bx、By 和 Bz 列名
        columns.append(f'{sensors[i]}_{sensors[j]}_Bx')
        columns.append(f'{sensors[i]}_{sensors[j]}_By')
        columns.append(f'{sensors[i]}_{sensors[j]}_Bz')



# 提取相应的列并生成训练集
X_test = test_df[columns].values

# X_test = test_df[[f'sensor_{i+1}_Bx' for i in range(8) if i != 6] + 
#                   [f'sensor_{i+1}_By' for i in range(8) if i != 6] + 
#                   [f'sensor_{i+1}_Bz' for i in range(8) if i != 6]].values

# row_mins = X_test.min(axis=1, keepdims=True)
# row_maxs = X_test.max(axis=1, keepdims=True)
# X_row_normalized = (X_test - row_mins) / (row_maxs - row_mins + 1e-8)  # 防止分母为 0
X_test_tensor = torch.FloatTensor(X_test).to(device)

# 进行预测
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.cpu().numpy()


# Ground truth 和预测数据
ground_truth = test_df[['magnet_x', 'magnet_y', 'magnet_z']].values
predicted = y_pred[:, :3]  # 只取位置部分


ground_truth_orientation = test_df[['theta', 'phi']].values
orientation_pred = y_pred[:, 3:6]  # 只取方向部分
# print(orientation_pred)

# ======================
# 位置误差（Euclidean distance）
# ======================
pos_error = np.linalg.norm(predicted - ground_truth, axis=1)  # shape: (464,)
mean_pos_error = np.mean(pos_error)
print(f"Mean Position Error (Euclidean Distance): {mean_pos_error:.4f}")

# ======================
# 方向误差（角度偏差）
# ======================

# ground truth θ/phi
theta_gt = ground_truth_orientation[:, 0]
phi_gt = ground_truth_orientation[:, 1]

# predicted θ/phi
theta_pred = orientation_pred[:, 0]
phi_pred = orientation_pred[:, 1]

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