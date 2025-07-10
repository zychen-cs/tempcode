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

from Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from SelfAttention_Family import FullAttention, AttentionLayer
from embed1 import DataEmbedding
import numpy as np



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

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
    
class InConv2D(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_ch, kernel_size=(3, 3), padding=(1, 1)),  # ������ 1 ͨ��
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):  # x: (B, 3, 28)
        x = x.unsqueeze(1)  # �� (B, 1, 3, 28)
        x = self.conv(x)    # �� (B, out_ch, 3, 28)
        x = x.flatten(2)    # �� (B, out_ch, 84)
        return x            # ������״���� 1D InConv ����� (B, d_channel, L=84)
class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, d_model, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, out_ch, kernel_size=1)

    def forward(self, x):  # x: [B, d_model, T]
        out = self.conv1(x)
        return out
class ConvTransformer(nn.Module):

    def __init__(self,enc_in, c_out, d_channel=64, d_model=32, n_heads=8, e_layers=3, d_ff=512, factor=1,
                 dropout=0.0, activation='gelu',output_attention=True):
        super(ConvTransformer, self).__init__()
        self.output_attention = output_attention
        
        # Embedding
        self.channel_embedding = InConv(3, d_channel)
        # self.channel_embedding = InConv2D(d_channel)
        self.enc_embedding = DataEmbedding(enc_in,d_model,dropout)
        # self.enc_embedding = DataEmbedding(84,d_model,dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.channel_projection = OutConv(d_channel, 3)
        # self.channel_projection = OutConv(c_out, 3)
        self.projection = nn.Linear(d_model,c_out, bias=True)


        self.reconstruction = OutConv(d_channel, 3)  # (B, c_out, L)
        self.proj_in = nn.Conv1d(d_channel, d_model, kernel_size=1)  # [B, d_model, L]
        self.proj_out = nn.Conv1d(d_model, d_channel, kernel_size=1)  # [B, d_channel, L]
        self.decoder = OutConv(d_channel, 28)
    def anomaly_detection(self, x_enc):
        # Embedding
        x_enc=self.channel_embedding(x_enc) # B,d_channel,L
        # x_enc=self.proj_in(x_enc)  # [B, d_model, L] �´���
        # x_enc = x_enc.permute(0, 2, 1)  # �� [B, L, d_model]
       

        enc_out = self.enc_embedding(x_enc)   # B,d_channel,d_model �ɴ�����Ҫ
         
      
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # x = x.permute(0, 2, 1)  # [B, d_model, L]

        # x = self.proj_out(x)  # [B, d_channel, L]
        # x = self.decoder(x)  # [B, out_ch=3, L]
        # print(enc_out.shape)
        print(enc_out.shape)
        enc_out=F.relu(self.channel_projection(enc_out))
        dec_out = self.projection(enc_out)

        # print(x.shape)
        return dec_out 

    def forward(self, x_enc):  # x_enc: (B, 3, 28)
        # x_enc = x_enc.permute(0, 2, 1) 
        dec_out = self.anomaly_detection(x_enc)
        return dec_out



# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size=128
d_channel=3
d_ff=1024
# d_model=256
d_model=128
output_c=28
dropout=0.1
e_layers=3
input_c=28
n_heads=4
model = ConvTransformer(enc_in=input_c, c_out=output_c, e_layers=e_layers, d_channel=d_channel, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
# model.load_state_dict(torch.load("/home/czy/桌面/magx-main1/Codes/optimization/finetuning_ours.pth", map_location=torch.device('cpu')))  # 加载权重
model.load_state_dict(torch.load("/home/czy/桌面/magx-main1/Codes/optimization/0603ours.pth", map_location=torch.device('cpu')))  # 加载权重

# model.load_state_dict(torch.load("/home/czy/桌面/magx-main1/Codes/optimization/0418pretrained_encoder_v2.pth", map_location=torch.device('cpu')))  # 加载权重
model.to(device)
model.eval()



tracking_model = MLP().to(device)
# tracking_model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0301_differential_mlp_model_alldir_v1.pth',map_location=torch.device('cpu')))
tracking_model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0130_differential_mlp_model_v3.pth',map_location=torch.device('cpu')))
# tracking_model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0130_differential_mlp_model_v3.pth',map_location=torch.device('cpu')))

# Set the model to evaluation mode10
tracking_model.eval()

# 读取测试数据 
# test_df = pd.read_csv('/home/wangmingke/czy/magnetic/benchdataset/combined_differential_dataset_1cm.csv')

# 读取测试数据
test_df = pd.read_csv('/home/czy/桌面/magx-main1/thirdx_test.csv')
# test_df = pd.read_csv('/home/czy/桌面/magx-main1/normaldata/alltest.csv')
# 筛选出 theta=0.0 且 phi=0.0 的数据
test_df = test_df[(test_df['theta'] == 0.0) & (test_df['phi'] == 0.0)]
print(len(test_df))

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
print(X_test.shape)

# Step 1: reshape to (batch_size, channels, sensors)
X_recon_input = X_test.reshape(X_test.shape[0], 28, 3).transpose(0, 2, 1)  # shape: (464, 3, 28)
print("Input to ConvTransformer shape:", X_recon_input.shape)

# Step 2: 转为 tensor，并输入到 ConvTransformer 模型
X_recon_input_tensor = torch.tensor(X_recon_input, dtype=torch.float32).to(device)
with torch.no_grad():
    X_denoised = model(X_recon_input_tensor)  # output shape: (464, 3, 28)

# Step 3: reshape back to (464, 84)
X_denoised_np = X_denoised.cpu().numpy().transpose(0, 2, 1).reshape(X_test.shape[0], -1)  # shape: (464, 84)
print("Input to tracking model shape:", X_denoised_np.shape)

# Step 4: convert to tensor and input to tracking model
X_denoised_tensor = torch.tensor(X_denoised_np, dtype=torch.float32).to(device)
tracking_output=[]
with torch.no_grad():
    tracking_output = tracking_model(X_denoised_tensor)

# tracking_output is the result from the tracking model
print("Tracking output shape:", tracking_output.shape)



# X_test_tensor = torch.FloatTensor(X_test).to(device)

# # 进行预测
# with torch.no_grad():
#     y_pred_tensor = model(X_test_tensor)
#     y_pred = y_pred_tensor.cpu().numpy()


# Ground truth 和预测数据
ground_truth = test_df[['magnet_x', 'magnet_y', 'magnet_z']].values
predicted = tracking_output[:, :3]  # 只取位置部分

ground_truth_orientation = test_df[['theta', 'phi']].values
orientation_pred = tracking_output[:, 3:6]  # 只取方向部分
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


# # 计算 position error（欧式距离）
# position_error = np.linalg.norm(ground_truth - predicted, axis=1)
# test_df['position_error'] = position_error

# # 计算样本的欧式距离
# euclidean_distance = np.linalg.norm(test_df[['magnet_x', 'magnet_y', 'magnet_z']], axis=1)

# # 定义距离区间
# # distance_bins = [5, 7, 9, 11,13]
# distance_bins = [5,6,7,8,9,10]
# position_errors_by_distance = [[] for _ in distance_bins]  # 初始化双重列表

# # 遍历所有样本，将 position error 分类到对应的距离区间
# # for i, dist in enumerate(euclidean_distance):
# #     error = position_error[i]
# #     for j, bin_edge in enumerate(distance_bins):
# #         if (dist <= bin_edge+0.1) & (dist >= bin_edge-0.1) :
# #             position_errors_by_distance[j].append(error)
# #             break  # 确保一个样本只进入一个区间
# # print(np.mean(position_errors_by_distance[1]))
# # print(np.mean(position_errors_by_distance[2]))
# # print(np.mean(position_errors_by_distance[5]))
# # print(position_errors_by_distance[6])
# for i, dist in enumerate(euclidean_distance):
#     error = position_error[i]
#     for j, bin_edge in enumerate(distance_bins):
#         if dist <= bin_edge:
#             position_errors_by_distance[j].append(error)
#             break  # 确保一个样本只进入一个区间

# # 计算每个距离区间的误差均值
# average_position_errors = [np.mean(errors) if errors else 0 for errors in position_errors_by_distance]
# data = {
#     'Distance (cm)': distance_bins,
#     'Average Position Error (cm)': average_position_errors
# }

# # 将字典转换为 DataFrame
# df = pd.DataFrame(data)

# # 保存到 CSV 文件
# # df.to_csv('average_position_errors_mlp_dataset5.csv', index=False)
# # # 绘制柱状图
# # x = np.arange(len(distance_bins))
# # fig, ax = plt.subplots(figsize=(8, 6))

# # # ax.bar(x, average_position_errors, color='#5D3A9B', alpha=0.7, label='Position Error')
# # ax.bar(x, average_position_errors, color='#E69F00', alpha=0.7, label='Position Error')

# # ax.set_xticks(x)
# # # 设置坐标轴字体大小
# # ax.set_xticks(x)
# # ax.set_xticklabels([f'{d}' for d in distance_bins], fontsize=25)
# # ax.set_yticklabels(ax.get_yticks(), fontsize=25)
# # # 设置轴标签
# # ax.set_xlabel('Distance (cm)', fontsize=25, labelpad=10)
# # ax.set_ylabel('Error (cm)', fontsize=25, labelpad=10)
# # ax = plt.gca()
# # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))  # 保留 1 位小数

# # # ax.set_ylabel('Error (cm)')
# # # ax.set_xlabel('Distance (cm)')
# # # ax.legend(fontsize=20)

# # plt.tight_layout()
# # plt.savefig("errors_by_euclidean_distance_mlp.png")
# # plt.show()


# x = np.arange(len(distance_bins))
# fig, ax = plt.subplots(figsize=(8, 6))

# ax.bar(x, average_position_errors, color='#E69F00', alpha=0.7, label='Position Error')

# ax.set_xticks(x)
# ax.set_xticklabels([f'{d}' for d in distance_bins], fontsize=25)
# ax.set_xlabel('Distance (cm)', fontsize=25, labelpad=10)

# # 设置 y 轴刻度
# y_ticks = np.arange(0, max(average_position_errors) + 0.05, 0.1)  # 从 0 到最大值，步长为 1
# ax.set_yticks(y_ticks)
# ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=25)

# ax.set_ylabel('Error (cm)', fontsize=25, labelpad=10)
# ax.legend(fontsize=20)
# plt.grid(axis="y")
# plt.tight_layout()
# plt.savefig('Figure4_a.jpg', dpi=300)  # 保存插值结果图像为 PDF 格式
# # plt.savefig("errors_by_euclidean_distance_mlp_lessz5.png")
# plt.show()