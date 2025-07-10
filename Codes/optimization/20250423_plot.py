import numpy as np
# 每一行是一个点，列为 x, y, z
import pandas as pd
data3 = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0423dynamic_static_3_1.csv")
data4 = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0423dynamic_static_4_1.csv")
mean_point3 = data3[['x', 'y', 'z']].iloc[:300].mean()
mean_point4 = data4[['x', 'y', 'z']].iloc[:300].mean()

data = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0423reconstruct_dynamic_static_3_1.csv")
data1 = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0423reconstruct_dynamic_static_4_1.csv")
mean_point1 = data[['x', 'y', 'z']].iloc[:300].mean()
mean_point2 = data1[['x', 'y', 'z']].iloc[:300].mean()

points1 = np.array([[0.020236775,-4.8857613,0.73584694], [-0.04694113,-6.772611,0.7911734]])
points2 = np.array([[0.23078242,-4.8688607,	0.796564], [0.53860617,-6.785567,0.84523207]])
points = np.array([[0,-5,0.7], [0,-7,0.7]])
# 逐点欧式距离

points_wo = np.array([[-1.011699,-4.915168,1.0004206], [-3.090248,-5.637272,0.8749806]])
points_w = np.array([mean_point1, mean_point2])

error = np.linalg.norm(points1 - points, axis=1)
error1 = np.linalg.norm(points2 - points, axis=1)


points_wo = np.array([mean_point3, mean_point4])
points_w = np.array([mean_point1, mean_point2])
error = np.linalg.norm(points_wo - points, axis=1)
error1 = np.linalg.norm(points_w - points, axis=1)

print("欧式距离 error:", error)
print("欧式距离 error:", error1)
