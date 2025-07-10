# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *


#模型1 CNN
class CNNEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        # x: [B, 28, 3]
        x = x.permute(0, 2, 1)  # => [B, 3, 28]

        x = self.conv1(x)       # => [B, 64, 28]
        x = self.conv2(x)       # => [B, 128, 28]

        # Residual block 1
        res = x
        x = self.conv3(x)
        x = x + res             # residual connection
        x = nn.ReLU()(x)        # 添加额外非线性激活（可选）

        # Residual block 2
        res = x
        x = self.conv4(x)
        x = x + res
        x = nn.ReLU()(x)

        x = self.pool(x).squeeze(-1)  # => [B, 128]
        x = self.fc(x)                # => [B, out_dim]
        return x







class Decoder(nn.Module):
    def __init__(self, input_dim=128, output_dim=84):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# class ClassifierHead(nn.Module):
#     def __init__(self, input_dim=128, num_classes=3):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):
#         return self.fc(x)  # logits

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)





#模型2 MLP
class MLPEncoder(nn.Module):
    def __init__(self, input_dim=84, hidden_dims=[256, 128], out_dim=128):
        super(MLPEncoder, self).__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.3)
        )

        self.res_block = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.3)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.3)
        )

        self.out = nn.Linear(hidden_dims[1], out_dim)

    def forward(self, x):
        x = self.flatten(x)              # [B, 84]
        x = self.fc1(x)                  # [B, 256]

        res = x                          # 残差输入
        x = self.res_block(x) + res     # 残差连接
        x = nn.ReLU()(x)                # 可选：额外激活

        x = self.fc2(x)                 # [B, 128]
        x = self.out(x)                 # [B, out_dim]
        return x


#模型3 Unet
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x=x.unsqueeze(-1).permute(0,2,1)  # [B, 1, 84]
        # print("Input:", x.shape)  # [B, 1, 84]
        x1 = self.inc(x)
        # print("x1:", x1.shape)

        x2 = self.down1(x1)
        # print("x2:", x2.shape)

        x3 = self.down2(x2)
        # print("x3:", x3.shape)

        x4 = self.down3(x3)
        # print("x4:", x4.shape)

        x5 = self.down4(x4)
        # print("x5:", x5.shape)

        x = self.up1(x5, x4)
        # print("up1:", x.shape)

        x = self.up2(x, x3)
        # print("up2:", x.shape)

        x = self.up3(x, x2)
        # print("up3:", x.shape)

        x = self.up4(x, x1)
        # print("up4:", x.shape)

        x = self.outc(x)
        # print("outc:", x.shape)

        return x.squeeze(1)  # [B, 84]


