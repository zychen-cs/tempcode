import pandas as pd
import numpy as np
import os

# 读取原始数据
# input_file = '/home/wangmingke/czy/magnetic/benchdataset/dataset_0cm.csv'
# df = pd.read_csv(input_file)

psensor_circular = 1e-2 * np.array([
[-0.6,0,0],
[0.6,0,0],
[0,-0.5,0],
[0.9,0.5,0],
[0.3,0.5,0],
[0,0,0],
[-0.3,0.5,0],
[-0.9,0.5,0]

])


# 传感器阵列位置
sensors = 1e-2 * np.array([
    [0.5, -0.5, 0],   # 5
    [-1, 0, 0],      # 25
    [-0.5, -0.5, 0], # 7
    [0, -0.8, 0],    # 4
    [1, 0, 0],       # 2
    [0.5, 0.5, 0],   # 15
    [0, 0, 0],       # 11
    [-0.5, 0.5, 0],  # 16
])



# 初始化结果字典
results = {'magnet_x': [], 'magnet_y': [], 'magnet_z': [], 'theta': [], 'phi': []}
for i in range(len(sensors)):
    results[f'sensor_{i+1}_Bx'] = []
    results[f'sensor_{i+1}_By'] = []
    results[f'sensor_{i+1}_Bz'] = []

# 计算合成磁场数据

# 提取磁铁位置和方向
magnet_x = 2.5
magnet_y = 4.5
magnet_z = 2
theta = np.pi/2
phi = 0

# 计算磁场
VecM = np.array([
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta)
]) * 1e-7 * np.exp(np.log(0.46))  # 假设 m 值为 0.46

for i, sensor in enumerate(sensors):
    VecR = 1e-2 * np.array([magnet_x, magnet_y, magnet_z]) - sensor  # 计算磁铁与传感器的向量
    NormR = np.linalg.norm(VecR)  # 计算磁铁与传感器之间的距离
    B = (3.0 * VecR * (np.dot(VecM, VecR)) / NormR**5 - VecM / NormR**3)  # 计算磁场分量
    print(VecR)
    print(NormR)
    print(B[0] * 1e6)  # 转换为 uT
    print(B[1] * 1e6)  # 转换为 uT
    print(B[2] * 1e6)  # 转换为 uT


