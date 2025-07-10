import numpy as np

# 加载NPZ文件
data = np.load('/home/czy/桌面/magx-main1/result/calibration.npz')
features = data['scale']
offset = data['offset']

# 使用数据（例如，打印或进行计算）
print("scale:", features)
print("offset:", offset)
# data是一个类似字典的对象
