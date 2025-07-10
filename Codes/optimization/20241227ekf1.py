import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 卡尔曼滤波器类
class KalmanFilter:
    def __init__(self, process_noise, measurement_noise, initial_estimate, initial_covariance):
        self.process_noise = process_noise  # 过程噪声协方差
        self.measurement_noise = measurement_noise  # 测量噪声协方差
        self.estimate = initial_estimate  # 初始状态估计
        self.covariance = initial_covariance  # 初始估计误差协方差

    def update(self, measurement):
        # 预测步骤
        predicted_estimate = self.estimate
        predicted_covariance = self.covariance + self.process_noise

        # 更新步骤
        kalman_gain = predicted_covariance / (predicted_covariance + self.measurement_noise)
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self.covariance = (1 - kalman_gain) * predicted_covariance

        return self.estimate

# 假设你已经有8个传感器的数据，每个传感器有3个轴（x, y, z），共24个轴
def initialize_filters():
    filters = {}
    # 假设每个传感器每个轴的噪声参数
    process_noise = 1e-3
    measurement_noise = 0.74
    initial_estimate = 0
    initial_covariance = 1
    
    for sensor_id in range(1, 9):  # 8个传感器
        for axis in ['x', 'y', 'z']:  # 每个传感器的3个轴
            # 为每个轴创建一个卡尔曼滤波器实例
            filters[f'sensor_{sensor_id}_{axis}'] = KalmanFilter(process_noise, measurement_noise, initial_estimate, initial_covariance)
    return filters

# 创建卡尔曼滤波器实例
kf_filters = initialize_filters()

# 读取传感器数据（假设你已经有一个CSV文件）
data = pd.read_csv("/media/czy/T7 Shield/ubuntu/calibration_project/differential/stdnoise.csv")
# 假设数据中包含24列：sensor_1_Bx, sensor_1_By, sensor_1_Bz, ..., sensor_8_Bz

time = []
noisy_signals = {}

# 假设你需要处理sensor_1到sensor_8的3个轴
for sensor_id in range(1, 9):
    noisy_signals[sensor_id] = {
        'x': data[f'sensor_{sensor_id}_Bx'],
        'y': data[f'sensor_{sensor_id}_By'],
        'z': data[f'sensor_{sensor_id}_Bz']
    }

# 使用卡尔曼滤波器去噪
filtered_signals = {sensor_id: {'x': [], 'y': [], 'z': []} for sensor_id in range(1, 9)}

# 假设数据每个点的时间间隔为1/15秒
for i in range(len(data)):
    time.append(i / 15)
    
    # 对每个传感器的每个轴进行滤波
    for sensor_id in range(1, 9):
        for axis in ['x', 'y', 'z']:
            measurement = noisy_signals[sensor_id][axis].iloc[i]
            filtered_value = kf_filters[f'sensor_{sensor_id}_{axis}'].update(measurement)
            filtered_signals[sensor_id][axis].append(filtered_value)

# 绘制去噪后的信号
plt.figure(figsize=(12, 8))

# 绘制所有传感器的x轴去噪信号（以sensor_1为例）
plt.plot(time, filtered_signals[2]['y'], label='Filtered sensor_1_x', linewidth=2)

# 绘制原始信号（带噪声）
plt.plot(time, noisy_signals[2]['y'], label='Noisy sensor_1_x', color='orange', alpha=0.6)

plt.title('Sensor Signal Denoising using Kalman Filter')
plt.xlabel('Time (seconds)')
plt.ylabel('Signal Value')
plt.legend()
plt.grid(True)
plt.show()
