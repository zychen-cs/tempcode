import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 真空磁导率
mu_0 = 4 * np.pi * 1e-7

# 计算磁矩的分量
def magnetic_moment(m, theta, phi):
    # 将 m 从球坐标系转换到直角坐标系
    mx = np.sin(theta) * np.cos(phi)  # X 轴分量
    my = np.sin(theta) * np.sin(phi)  # Y 轴分量
    mz = np.cos(theta)               # Z 轴分量
    return np.array([mx, my, mz])

# 磁偶极子模型
def magnetic_field_dipole(m, r_vec):
    r = np.linalg.norm(r_vec)  # 计算 r 的模
    return (mu_0 / (4 * np.pi)) * (3 * np.dot(m, r_vec) * r_vec - m * r**2) / r**5

def calculation_parallel():
    global result_draw

    # 读取数据
    data = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0911sensor_2.csv")
    num = 8  # 传感器数量

    # 传感器数据初始化
    sensors = np.zeros((num, 3))
    calibration = np.load('result/calibration.npz')
    offset = calibration['offset'].reshape(-1)
    scale = calibration['scale'].reshape(-1)

    # 磁铁参数 (dipole model)
    # 0.022901412095202	2.43858448443522

    theta = 0.02  # 给定的 theta
    phi = 2.44    # 给定的 phi
    # theta = 3.12  # 给定的 theta
    # phi = 0.69    # 给定的 phi
    m_magnitude = 0.25  # 磁矩的大小
    m = magnetic_moment(m_magnitude, theta, phi)  # 计算磁矩在 x, y, z 轴的分量
    # r_vec = np.array([-0.54, -4.76, 0.13])  # 磁铁到传感器的距离向量
    r_vec = np.array([3.04, -2.17, 0.18])
    dis = -0.54  # 计算 scale_factor=0 时的距离
    print(f"Initial Distance (scale_factor=0): {dis:.4f} meters")

    plt.figure(figsize=(10, 6))  # 创建一张图

    # **1. 先计算 scale_factor=0 的情况**
    scale_factor_0 = 0
    result_draw_0 = []
    modified_scale_0 = scale + scale_factor_0  # 修改 scale
    for j in range(len(data)):
        sensors = np.zeros((num, 3))
        for i in range(num):
            column_name = f"Sensor {i+1}"
            datatemp = data[column_name][j]
            datatemp = datatemp.strip("()")
            numbers = datatemp.split(", ")

            sensors[i, 0] = float(numbers[0])
            sensors[i, 1] = float(numbers[1])
            sensors[i, 2] = float(numbers[2])

        sensors = sensors.reshape(-1)
        sensors = (sensors - offset) / modified_scale_0 * 33
        # 计算磁场强度并存储结果
        result_draw_0.append(math.sqrt((sensors[18]**2 + sensors[19]**2 + sensors[20]**2)-32)*m[0])

    # 计算 scale_factor=0 时的平均磁场强度
    avg_magnetic_field_0 = np.mean(result_draw_0)
    
    print(f"Scale {scale_factor_0}: Set baseline magnetic field B0 = {avg_magnetic_field_0:.4f} uT")

    # 存储每个 scale_factor 下的距离
    distances = []  # scale_factor=0 的初始距离

    # **2. 计算其他 scale_factor 的情况**
    scale_factors = range(-10, 11, 1)  # 所有的 scale_factor
    for scale_factor in scale_factors:
        result_draw = []
        modified_scale = scale + scale_factor  # 修改 scale
        for j in range(len(data)):
            sensors = np.zeros((num, 3))
            for i in range(num):
                column_name = f"Sensor {i+1}"
                datatemp = data[column_name][j]
                datatemp = datatemp.strip("()")
                numbers = datatemp.split(", ")

                sensors[i, 0] = float(numbers[0])
                sensors[i, 1] = float(numbers[1])
                sensors[i, 2] = float(numbers[2])

            sensors = sensors.reshape(-1)
            sensors = (sensors - offset) / modified_scale * 33
            # 计算磁场强度并存储结果
            result_draw.append(math.sqrt((sensors[18]**2 + sensors[19]**2 + sensors[20]**2)-32)*m[0])

        # 计算当前 scale 下磁场强度的均值
        avg_magnetic_field = np.mean(result_draw)
        print(f"Scale {scale_factor}: Avg Magnetic Field = {avg_magnetic_field:.4f} uT")

        # 通过 B0 推算此时磁铁的距离
        new_distance = dis * (avg_magnetic_field_0 / avg_magnetic_field)**(1/3)
        distances.append(new_distance)
        print(f"Scale {scale_factor}: Estimated Distance = {new_distance:.4f} meters")

        # 绘制不同 scale 值的结果曲线
        if(scale_factor==0):
            plt.plot(result_draw_0, label=f'Scale {scale_factor_0} (Original)', color='red', linewidth=2.5)
        else:
            plt.plot(result_draw, label=f'Scale {scale_factor}', linewidth=1)

    # 设置图例、标题和标签
    plt.legend()
    plt.xlabel('Index', fontsize=20)
    plt.ylabel('Magnetic Field Intensity(uT)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 绘制距离与 scale_factor 的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(scale_factors, distances, marker='o', color='blue', linewidth=2)
    plt.xlabel('Scale Factor', fontsize=20)
    plt.ylabel('Estimated Distance (cm)', fontsize=20)
    # plt.title('Estimated Distance vs. Scale Factor', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    calculation_parallel()
