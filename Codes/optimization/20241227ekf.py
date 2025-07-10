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

# 设置卡尔曼滤波器的参数
process_noise = 1e-3  # 减小过程噪声，表示对预测的信任较大
measurement_noise = 0.23  # 增大测量噪声，表示对传感器的信任较小

initial_estimate = 0  # 初始状态估计
initial_covariance = 1  # 初始估计误差协方差

# 创建卡尔曼滤波器
kf = KalmanFilter(process_noise, measurement_noise, initial_estimate, initial_covariance)

# 生成合成的传感器数据（带噪声）
# 假设真实信号是正弦波加上随机噪声
# time = np.linspace(0, 10, 500)
# true_signal = np.sin(time)  # 真实信号
# noise = np.random.randn(len(time)) * 0.5  # 添加噪声
# print(np.std(noise))
time=[]
data = pd.read_csv("/media/czy/T7 Shield/ubuntu/calibration_project/differential/stdnoise.csv")
noisy_signal = data["sensor_1_Bx"]

# 使用卡尔曼滤波器去噪
filtered_signal = [kf.update(value) for value in noisy_signal]
for i in range(len(data['sensor_1_Bx'])):
    time.append(i/15)
# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制原始信号（带噪声）
plt.plot(time, noisy_signal, label='Noisy Signal', color='orange', alpha=0.6)

# 绘制去噪后的信号
plt.plot(time, filtered_signal, label='Filtered Signal', color='blue', linewidth=2)

# 绘制真实信号
# plt.plot(time, noisy_signal, label='True Signal', color='green', linestyle='--')

# 图例和标签
plt.title('Sensor Signal Denoising using Kalman Filter')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.legend()

# 显示图形
plt.grid(True)
plt.show()
