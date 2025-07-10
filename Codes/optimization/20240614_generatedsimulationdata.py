import numpy as np
import pandas as pd

# 传感器阵列位置
sensors = 1e-2 *np.array([
    [0, -0.8, 0],
    [0.5, -0.5, 0],
    [-0.5, -0.5, 0],
    [1, 0, 0],
    [0, 0, 0],
    [-1, 0, 0],
    [0.5, 0.5, 0],
    [-0.5, 0.5, 0],
    [1, 1, 0],
    [-1, 1, 0]
])

# 磁铁m值
m = 0.25

# 磁铁移动范围
# x_range = np.arange(-5, 5.05, 0.05)
# y_range = np.arange(-25, 25.05, 0.05)
# z_range = np.arange(-5, 5.05, 0.05)

x_range = np.arange(-5, 5, 1)
y_range = np.arange(-5, 0, 1)
z_range = np.arange(0, 2, 1)

# 初始化结果字典
results = {f'sensor_{i+1}_Bx': [] for i in range(len(sensors))}
results.update({f'sensor_{i+1}_By': [] for i in range(len(sensors))})
results.update({f'sensor_{i+1}_Bz': [] for i in range(len(sensors))})

# 计算磁场
for x in x_range:
    for y in y_range:
        for z in z_range:
            for i, sensor in enumerate(sensors):
                VecM = np.array([
                    np.sin(0) * np.cos(0),
                    np.sin(0) * np.sin(0),
                    np.cos(0)
                ]) * 1e-7 * np.exp(m)
                VecR = sensor - 1e-2*np.array([x, y, z])
                NormR = np.linalg.norm(VecR)
                scalar_part = 3.0 * (np.dot(VecM, VecR) / NormR**5)
                B = scalar_part * VecR - VecM / NormR**3
                
                results[f'sensor_{i+1}_Bx'].append(B[0] * 1e6)
                results[f'sensor_{i+1}_By'].append(B[1] * 1e6)
                results[f'sensor_{i+1}_Bz'].append(B[2] * 1e6)

# 将结果保存到CSV文件
df = pd.DataFrame(results)
df.to_csv('sensor_data.csv', index=False)
