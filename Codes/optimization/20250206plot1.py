import pandas as pd
import numpy as np

# 读取 CSV 文件
before1 = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0206MLP_before.csv")
before2 = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0206LM_before.csv")

# 设定基准值
ground_x = 0
ground_y = -5
ground_z = 0.7
ground_truth = np.array([ground_x, ground_y, ground_z])

# 计算欧式距离误差
def euclidean_distance(row, ground_truth):
    # print()
    return np.sqrt((row['x'] - ground_truth[0])**2 + (row['y'] - ground_truth[1])**2 + (row['z'] - ground_truth[2])**2)

# 对两个数据集计算欧式距离误差
errors_before1 = before1.apply(lambda row: euclidean_distance(row, ground_truth), axis=1)
errors_before2 = before2.apply(lambda row: euclidean_distance(row, ground_truth), axis=1)

# 计算均值和标准差
# print(errors_before1)
mean_error1 = errors_before1.mean()
std_error1 = errors_before1.std()

mean_error2 = errors_before2.mean()
std_error2 = errors_before2.std()

# 输出结果
print("Before1 (0206MLP_before.csv) 欧式距离误差统计:")
print(f"均值误差: {mean_error1}")
print(f"标准差: {std_error1}\n")

print("Before2 (0206LM_before.csv) 欧式距离误差统计:")
print(f"均值误差: {mean_error2}")
print(f"标准差: {std_error2}")
