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
from os import read
import queue
from codetiming import Timer
import asyncio
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from itertools import count
import time
from matplotlib.animation import FuncAnimation
from numpy.core.numeric import True_
import matplotlib
import queue
import asyncio
import struct
import os
import sys
import time
import datetime
import atexit
import time
import numpy as np
from bleak import BleakClient
import matplotlib.pyplot as plt
from bleak import exc
import pandas as pd
import atexit
from multiprocessing import Pool
import multiprocessing

from src.solver import Solver_jac, Solver
from src.filter import Magnet_KF, Magnet_UKF
from src.preprocess import Calibrate_Data
from config import pSensor_smt, pSensor_joint_exp,pSensor_large_smt, pSensor_small_smt, pSensor_median_smt, pSensor_imu, pSensor_ear_smt,pSensor_selfcare
import cppsolver as cs

pSensor = pSensor_joint_exp

params = np.array([np.log(0.46), 1e-2 * (-0.5), 1e-2 * (-7.5), 1e-2 * (3), 0.5, 0])
name = ['Time Stamp', 'x',
        'y', 'z', 'theta', 'phy']

countnum=1
countnum1=1
resultslist=[]
resultslist1=[]





global worklist

global params2
global results
global results2
# global direction
myparams1 = params

def ang_convert(x):
    a = x//(2*np.pi)
    result = x-a*(2*np.pi)
    # if result > np.pi:
    #     result -= np.pi * 2
    if result <0:
        result += np.pi * 2
    return result

def ang_convert1(x):
    a = x//(2*np.pi)
    result = x-a*(2*np.pi)
    if result > np.pi:
        result -= np.pi
    if result <0:
        result += np.pi
    return result

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
model.load_state_dict(torch.load("/home/czy/桌面/magx-main1/Codes/optimization/finetuning_ours.pth", map_location=torch.device('cpu')))  # 加载权重
# model.load_state_dict(torch.load("/home/czy/桌面/magx-main1/Codes/optimization/0418pretrained_encoder_v2.pth", map_location=torch.device('cpu')))  # 加载权重
model.to(device)
model.eval()



tracking_model = MLP().to(device)
# tracking_model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0130_differential_mlp_model_v3.pth',map_location=torch.device('cpu')))
# tracking_model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0130_differential_mlp_model_v3.pth',map_location=torch.device('cpu')))
tracking_model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0301_differential_mlp_model_alldir_v1.pth',map_location=torch.device('cpu')))
# Set the model to evaluation mode10
tracking_model.eval()

# 读取测试数据 
# test_df = pd.read_csv('/home/wangmingke/czy/magnetic/benchdataset/combined_differential_dataset_1cm.csv')

# 读取测试数据
# test_df = pd.read_csv('/home/czy/桌面/magx-main1/secondz_test.csv')
test_df = pd.read_csv('/media/czy/T7 Shield/mobicom_dataset/noisetype1/noise_z/firstz_test.csv')
# test_df = pd.read_csv('/home/czy/桌面/magx-main1/Codes/optimization/alltest_LMdiff.csv')
# 筛选出 theta=0.0 且 phi=0.0 的数据
test_df = test_df[(test_df['theta'] == 0.0) & (test_df['phi'] == 0.0)]
print(len(test_df))
# test_df = test_df.iloc[455:]  # 从第133行开始（索引从0开始）
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



X_denoised_np = X_test
all_data = []
worklist = multiprocessing.Manager().Queue()
magcount=1
results=[]
results = multiprocessing.Manager().Queue()
for j in range (0,len(X_test)):
        readings = X_denoised_np[j]  # shape: (3, 28)
        # readings = denoised_sample.transpose(1, 0).reshape(-1).tolist()
        # Step 3: reshape back to (464, 84)

        # print(readings)
        all_data.append(readings)
        worklist.put(readings)
        if not worklist.empty():
                datai = worklist.get()
                # datai = datai.reshape(-1, 3)
                # resulti [gx, gy, gz, m, x0,y0,z0, theta0, phy0, x1, y1, z1, theta1, phy1]
                if magcount == 1:
                    if np.max(np.abs(myparams1[1:4])) > 0.1:
                        myparams1 = params
                    
                    x_ground = test_df["magnet_x"][j]
                    y_ground = test_df["magnet_y"][j]
                    z_ground = test_df["magnet_z"][j]
                    theta_ground = test_df["theta"][j]
                    phi_ground = test_df["phi"][j]
                    myparams1 = np.array([np.log(0.46), 1e-2 * (x_ground), 1e-2 * (y_ground), 1e-2 * (z_ground), theta_ground, phi_ground])
                    # myparams1 = np.array([np.log(0.46), 1e-2 * (x_ground), 1e-2 * (y_ground), 1e-2 * (z_ground), theta_ground, phi_ground])
                    # print(myparams1)
                    resulti = cs.solve_1mag(
                        datai, pSensor.reshape(-1), myparams1)
                    # myparams1 = resulti
                    
                    result = [resulti[1] * 1e2,
                            resulti[2] * 1e2, resulti[3] * 1e2]
                    results.put(result)
                    current = [datetime.datetime.now()]
                    # direction=np.array([np.sin(ang_convert(resulti[7]))*np.cos(ang_convert(resulti[8])),
                    #           np.sin(ang_convert(resulti[7]))*np.sin(ang_convert(resulti[8])), np.cos(ang_convert(resulti[7]))])
                    current.append(resulti[1] * 1e2)
                    current.append(resulti[2] * 1e2)
                    current.append(resulti[3] * 1e2)
                    current.append(ang_convert1(resulti[4]))
                    current.append(ang_convert(resulti[5]))
                    # current.append((resulti[7]))
                    # current.append((resulti[8]))
                    resultslist.append(current)
                    # print(resultslist)
                    # resultlist.append(current)
                    # print(resulti[3])
                    
                    # print("Orientation: {:.2f}, {:.2f}, m={:.2f}".format(
                    #     resulti[4] / np.pi * 180,
                    #     resulti[5] / np.pi * 180 % 360,
                    #     np.exp(resulti[0])))
                    # print("Position: {:.2f}, {:.2f}, {:.2f}, m={:.2f}, dis={:.2f},orientation: {:.2f},{:.2f}".format(
                    #     result[0],
                    #     result[1],
                    #     result[2],
                    #     resulti[0],
                    #     np.sqrt(
                    #         result[0] ** 2 + result[1] ** 2 + result[2] ** 2),
                    #     ang_convert1(resulti[4]),
                    #     ang_convert(resulti[5]))),

                    # print(data["magnet_x"][0])
                    # print(data["magnet_y"][0])
                    # print(data["magnet_z"][0])
print("Output csv")
test = pd.DataFrame(columns=name,data=resultslist)
# test.to_csv("/home/czy/桌面/magx-main1/Codes/optimization/ours_LMDiff_noise2_firstx.csv")
print("Exited") 

# X_denoised_np = X_denoised.cpu().numpy().transpose(0, 2, 1).reshape(X_test.shape[0], -1)  # shape: (464, 84)
# print("Input to tracking model shape:", X_denoised_np.shape)

# # Step 4: convert to tensor and input to tracking model
# X_denoised_tensor = torch.tensor(X_denoised_np, dtype=torch.float32).to(device)
# tracking_output=[]
# with torch.no_grad():
#     tracking_output = tracking_model(X_denoised_tensor)

# # tracking_output is the result from the tracking model
# print("Tracking output shape:", tracking_output.shape)



# # Ground truth 和预测数据
ground_truth = test_df[['magnet_x', 'magnet_y', 'magnet_z']].values
predicted = test[['x', 'y', 'z']].values  # 只取位置部分

ground_truth_orientation = test_df[['theta', 'phi']].values
orientation_pred = test[['theta', 'phy']].values
# orientation_pred = test[:, 4:7]  # 只取方向部分
# print(orientation_pred)

# # ======================
# # 位置误差（Euclidean distance）
# # ======================
pos_error = np.linalg.norm(predicted - ground_truth, axis=1)  # shape: (464,)
mean_pos_error = np.mean(pos_error)
print(f"Mean Position Error (Euclidean Distance): {mean_pos_error:.4f}")

# # ======================
# # 方向误差（角度偏差）
# # ======================

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


