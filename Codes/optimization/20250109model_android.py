import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
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



# 加载训练好的模型
model = MLP()
model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0127_differential_mlp_model_interpolation_v3.pth',map_location=torch.device('cpu')))
# model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0108_differential_mlp_model_v3.pth'))
model.eval()

# 将模型转换为TorchScript格式
scripted_model = torch.jit.script(model)  # 或者使用torch.jit.trace()
scripted_model.save('/home/czy/桌面/magx-main1/Codes/optimization/finalmodel1.pt')  # 保存为 model.pt 格式
