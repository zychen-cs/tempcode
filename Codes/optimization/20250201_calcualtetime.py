import os
import pandas as pd

# 定义文件夹路径
folder_path = '/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/'  # 替换为你的文件夹路径

# 存储所有 runtime 数据的列表
runtime_values = []

# 遍历文件夹中的文件
for letter in range(ord('A'), ord('Z') + 1):  # 遍历从 A 到 Z
    filename = f'0216MLP_{chr(letter)}_1.csv'  # 构造文件名
    file_path = os.path.join(folder_path, filename)  # 完整文件路径
    
    if os.path.exists(file_path):  # 确保文件存在
        # 读取 CSV 文件
        data = pd.read_csv(file_path)
        
        # 提取 'runtime' 列数据
        if 'runtime' in data.columns:  # 确保 'runtime' 列存在
            runtime_values.extend(data['runtime'].tolist())  # 将数据添加到列表中

# 计算所有文件中 runtime 数据的平均值
if runtime_values:
    average_runtime = sum(runtime_values) / len(runtime_values)
    print(f'所有文件中 runtime 列的平均值为: {average_runtime}')
else:
    print('没有找到任何 runtime 数据')
