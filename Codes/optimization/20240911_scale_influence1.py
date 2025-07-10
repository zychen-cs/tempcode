import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def calculation_parallel():
    global result_draw

    data = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0913sensor_2.csv")
    num = 8  # Sensor数量

    all_data = []
    sensors = np.zeros((num, 3))
    calibration = np.load('result/calibration.npz')
    offset = calibration['offset'].reshape(-1)
    scale = calibration['scale'].reshape(-1)

    plt.figure(figsize=(10, 6))  # 创建一张图

    # 遍历 scale 从 -10 到 10 的变化
    for scale_factor in range(-10, 11, 1):
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
            # result_draw.append(sensors[19])
            result_draw.append(math.sqrt(sensors[18]**2 + sensors[19]**2 + sensors[20]**2))
        if scale_factor == 0:
            plt.plot(result_draw, label=f'Scale {scale_factor} (Original)', color='red', linewidth=2.5)
        else:
            plt.plot(result_draw, label=f'Scale {scale_factor}', linewidth=1)
        # 绘制不同 scale 值的结果曲线
        # plt.plot(result_draw, label=f'Scale {scale_factor}')

    # 设置图例、标题和标签
    plt.legend()
    # plt.title('Effect of Scale on Magnetic Field Intensity')
    plt.xlabel('Index', fontsize=20)
    plt.ylabel('Magnetic Field Intensity(uT)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    calculation_parallel()
