import numpy as np
import pandas as pd
import time
import os

# 传感器阵列位置
sensors = 1e-2 * np.array([
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

# 磁铁 m 值
m = 0.25

# 磁铁移动范围
x_range = np.arange(-10, 10.05, 0.05)
y_range = np.arange(-25, -2.05, 0.05)
z_range = np.arange(-2, 4.05, 0.05)

# θ 和 φ 的范围
theta_range = np.arange(0, np.pi, np.pi / 36)  # 0 到 π，步长为 π/12
phi_range = np.arange(0, 2 * np.pi, np.pi / 18)  # 0 到 2π，步长为 π/9

# 初始化结果字典
results = {'magnet_x': [], 'magnet_y': [], 'magnet_z': [], 'theta': [], 'phi': []}
for i in range(len(sensors)):
    results[f'sensor_{i+1}_Bx'] = []
    results[f'sensor_{i+1}_By'] = []
    results[f'sensor_{i+1}_Bz'] = []

# 记录开始时间
start_time = time.time()
print(start_time)

# 计算磁场并分批保存
batch_size = 10000  # 每次保存的行数
batch_counter = 0

output_file = 'sensor_data.csv'

# 删除已存在的文件，以确保每次运行时都是新的文件
if os.path.exists(output_file):
    os.remove(output_file)

for x in x_range:
    for y in y_range:
        for z in z_range:
            for theta in theta_range:
                for phi in phi_range:
                    for i, sensor in enumerate(sensors):
                        VecM = np.array([
                            np.sin(theta) * np.cos(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(theta)
                        ]) * 1e-7 * np.exp(m)
                        VecR = sensor - 1e-2 * np.array([x, y, z])
                        NormR = np.linalg.norm(VecR)
                        scalar_part = 3.0 * (np.dot(VecM, VecR) / NormR**5)
                        B = scalar_part * VecR - VecM / NormR**3
                        
                        results[f'sensor_{i+1}_Bx'].append(B[0] * 1e6)
                        results[f'sensor_{i+1}_By'].append(B[1] * 1e6)
                        results[f'sensor_{i+1}_Bz'].append(B[2] * 1e6)
                    
                    # 保存磁铁的位置和方向
                    results['magnet_x'].append(x)
                    results['magnet_y'].append(y)
                    results['magnet_z'].append(z)
                    results['theta'].append(theta)
                    results['phi'].append(phi)

                    batch_counter += 1

                    # 检查是否需要保存批处理结果
                    if batch_counter >= batch_size:
                        df = pd.DataFrame(results)
                        header = not os.path.exists(output_file)  # 如果文件不存在，则写入列名
                        df.to_csv(output_file, mode='a', header=header, index=False, float_format='%.2f')
                        for key in results:
                            results[key].clear()  # 清空结果字典以释放内存
                        batch_counter = 0

# 保存最后一批结果
if batch_counter > 0:
    df = pd.DataFrame(results)
    header = not os.path.exists(output_file)  # 如果文件不存在，则写入列名
    df.to_csv(output_file, mode='a', header=header, index=False, float_format='%.2f')

# 记录结束时间
end_time = time.time()

# 计算并打印运算时间
elapsed_time = end_time - start_time
print(f"Total computation time: {elapsed_time:.2f} seconds")
