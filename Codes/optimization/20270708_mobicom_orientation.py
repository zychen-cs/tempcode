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
class ConvTransformer1(nn.Module):

    def __init__(self,enc_in, c_out, d_channel=64, d_model=32, n_heads=8, e_layers=3, d_ff=512, factor=1,
                 dropout=0.0, activation='gelu',output_attention=True):
        super(ConvTransformer1, self).__init__()
        self.output_attention = output_attention
        
        # Embedding
        # self.channel_embedding = InConv2D(d_channel)
        self.channel_embedding = InConv(1, d_channel)
        self.enc_embedding = DataEmbedding(enc_in,d_model,dropout)
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
        # self.channel_projection = OutConv(c_out, 3)
        self.channel_projection = OutConv(d_channel, 1)
        self.projection = nn.Linear(d_model,c_out, bias=True)

        self.reconstruction = OutConv(d_channel, 1)  # (B, c_out, L)
        self.proj_in = nn.Conv1d(d_channel, d_model, kernel_size=1)  # [B, d_model, L]
        self.proj_out = nn.Conv1d(d_model, d_channel, kernel_size=1)  # [B, d_channel, L]
        self.decoder = OutConv(d_channel, 3)
    def anomaly_detection(self, x_enc):
        # Embedding
        print(x_enc.shape)
        x_enc=self.channel_embedding(x_enc) # B,d_channel,L
        print(x_enc.shape)
        # x_enc=self.proj_in(x_enc)  # [B, d_model, L] �´���
        # x_enc = x_enc.permute(0, 2, 1)  # �� [B, L, d_model]
       

        enc_out = self.enc_embedding(x_enc)   # B,d_channel,d_model �ɴ�����Ҫ
         
      
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # x = x.permute(0, 2, 1)  # [B, d_model, L]

        # x = self.proj_out(x)  # [B, d_channel, L]
        # x = self.decoder(x)  # [B, out_ch=3, L]
        # print(enc_out.shape)
        enc_out=F.relu(self.channel_projection(enc_out))
        dec_out = self.projection(enc_out)

        # print(x.shape)
        return dec_out 

    def forward(self, x_enc):  # x_enc: (B, 3, 28)
        # x_enc = x_enc.permute(0, 2, 1) 
        x_enc=x_enc.unsqueeze(-1).permute(0,2,1)    # B,1,84
        # print(x_enc.shape)
        dec_out = self.anomaly_detection(x_enc)
        return dec_out.permute(0,2,1).squeeze(-1)
        # return dec_out

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size=128
d_channel=3
d_ff=1024
# d_model=256
d_model=128
output_c=3
dropout=0.1
e_layers=3
input_c=3
n_heads=4
model = ConvTransformer1(enc_in=input_c, c_out=output_c, e_layers=e_layers, d_channel=d_channel, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
model.load_state_dict(torch.load("/home/czy/桌面/magx-main1/Codes/optimization/0708orientation_fin1.pth", map_location=torch.device('cpu')))  # 加载权重
# model.load_state_dict(torch.load("/home/czy/桌面/magx-main1/Codes/optimization/0613finetune.pth", map_location=torch.device('cpu')))  # 加载权重
# model.load_state_dict(torch.load("/home/czy/桌面/magx-main1/Codes/optimization/0418pretrained_encoder_v2.pth", map_location=torch.device('cpu')))  # 加载权重
model.to(device)
model.eval()





# 读取测试数据 
# test_df = pd.read_csv('/home/wangmingke/czy/magnetic/benchdataset/combined_differential_dataset_1cm.csv')

# 读取测试数据
# test_df = pd.read_csv('/home/czy/桌面/magx-main1/secondz_test.csv')
# test_df = pd.read_csv('/home/czy/桌面/magx-main1/secondx_test.csv')
test_df = pd.read_csv('/home/czy/Downloads/testdata/testdata/new/2025-07-08_16-38-49/Compass.csv')
# 筛选出 theta=0.0 且 phi=0.0 的数据



# 初始化一个空的列表来存储列名
columns = []
columns.append('X')
columns.append('Y')
columns.append('Z')
# # 遍历所有传感器对
# for i in range(len(sensors)):
#     for j in range(i + 1, len(sensors)):
#         # 生成每对传感器的 Bx、By 和 Bz 列名
#         columns.append(f'{sensors[i]}_{sensors[j]}_Bx')
#         columns.append(f'{sensors[i]}_{sensors[j]}_By')
#         columns.append(f'{sensors[i]}_{sensors[j]}_Bz')


X_test = test_df[columns].values.astype(np.float32)  # shape: (N, 3)
print(X_test)
# print(X_test)
outputs = []

model.eval()
with torch.no_grad():
    for i in range(X_test.shape[0]):
        sample = torch.tensor(X_test[i]).unsqueeze(0)  # (1, 3)
        output = model(sample)  # e.g., (1, 3)
        outputs.append(output.squeeze(0).numpy())  # (3,)

X_denoised = np.stack(outputs)  # shape: (N, 3)
# 将输出结果转换为 DataFrame
df_denoised = pd.DataFrame(X_denoised, columns=['x_denoised', 'y_denoised', 'z_denoised'])

# 保存为 CSV 文件
output_path = "/home/czy/Downloads/testdata/testdata/denoise15102_new.csv"
df_denoised.to_csv(output_path, index=False)

print(f"Denoised data saved to {output_path}")