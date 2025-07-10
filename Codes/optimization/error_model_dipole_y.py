import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Here is the complete code including the imports, reading the CSV file, and processing the data to extract the sensor values.

import pandas as pd
import numpy as np
import re
coeff_sen = np.array([[70.13222736 ,65.34360469 ,116.64620222],
                                [ 91.4852902,   64.55655607 ,125.14642455],
                                [ 32.65798375,  65.27682017, 70.44891319],
                                [ 78.20043183,  66.6808322 , 70.33469614],
                                [71.29525642,  71.77189573, 74.75607821],
                                [ 62.54935307,  68.22916598 ,70.46668267],
                                [ 82.77259599,  73.38872962 , 56.93140818],
                                [ 68.04815503,  70.28364437 , 56.55348397],
                                [ 72.42981535,  68.38534121 , 65.44988918],
                                [ 71.22174515,  68.46519957 , 66.33085574]])
factorSen = 45.2 / (128 * 4096)
# Function to extract float values from a string within parentheses
def extract_values(s):
    return list(map(float, re.findall(r'\(([^()]+)\)', s)[0].split(', ')))

# Function to process a single row of sensor data
def process_row(row):
    row_sensor_values = []
    for sensor_data in row:
        sensor_values = extract_values(sensor_data)
        row_sensor_values.extend(sensor_values)
    return row_sensor_values

# Read the CSV file containing the sensor data
csv_file_path = '/home/czy/桌面/magx-main1/0114calib_3.csv'
sensor_data_df = pd.read_csv(csv_file_path)

# Apply the process_row function to each row of sensor data
# Excluding the first two columns which are 'Unnamed: 0' and 'Time Stamp'
sensor_columns = sensor_data_df.columns[2:]
# print(sensor_columns)
sensor_data_processed = sensor_data_df[sensor_columns].apply(process_row, axis=1)
# print(sensor_data_processed)
# Convert the series of lists into a 2D numpy array
sensor_data_np = np.array(sensor_data_processed.tolist())
# print(sensor_data_np)
column_means1 = np.mean(sensor_data_np, axis=0)
column_means = np.min(sensor_data_np, axis=0)
column_max = np.max(sensor_data_np, axis=0)
column_stds = np.std(sensor_data_np, axis=0)
print(column_max)
print(column_stds)
csv_file_path1 = '/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0114sensor2_1.csv'
sensor_data_df1 = pd.read_csv(csv_file_path1)

# Apply the process_row function to each row of sensor data
# Excluding the first two columns which are 'Unnamed: 0' and 'Time Stamp'
sensor_columns1 = sensor_data_df1.columns[2:]
sensor_data_processed1 = sensor_data_df1[sensor_columns1].apply(process_row, axis=1)

# Convert the series of lists into a 2D numpy array
sensor_data_np1 = np.array(sensor_data_processed1.tolist())
position = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0114test2_1.csv")
result=[]
result1=[]
result2=[]
result3=[]
result4=[]
result5=[]
result6=[]
result7=[]
result8=[]
result9=[]
# 4, 5, 7, 2, 11, 25, 15, 16, 3, 26
pSensor_joint_exp = 1e-2 * np.array([
    [0, -0.8, 0],

    [ 0.5, -0.5, 0],
    [-0.5, -0.5, 0],

    [ 1, 0, 0],
    [ 0, 0, 0],
    [-1, 0, 0],

    [ 0.5, 0.5, 0],
    [-0.5, 0.5, 0],

    [ 1, 1, 0],
    [-1, 1, 0]
])

relativepos=[]
relativepos.append(0)
distance=[]
SNRstd= 10 * np.log10((column_max[20]-column_means[20]) / abs(column_means[20]))
disID=0
snrID=0
disindex=0
snrindex=0
# SNRstd= 10 * np.log10(-column_stds[1] / abs(column_means[1]))
for i in range(0,len(sensor_data_np1)):
    if (i>0):
        relativepos.append(-(position["y"][i]-position["y"][i-1]))
        if(relativepos[i]>0.3 and disID==0):
            disID=disID+1
            disindex = i-1
    # tempdata = sensor_data_np1[i][1] * (1 + coeff_sen[i, 0] * (Temper - 35) * factorSen)
    if(column_means[1]>0):
        distance.append(-position["y"][i])
        # print(abs(sensor_data_np1[i][1]-column_means[1]))
        SNR= 10 * np.log10(abs(sensor_data_np1[i][1]-column_means[1]) / abs(column_means[1]))
        SNR1= 10 * np.log10(abs(sensor_data_np1[i][4]-column_means[4]) / abs(column_means[4]))
        SNR2= 10 * np.log10(abs(sensor_data_np1[i][7]-column_means[7]) / abs(column_means[7]))
        SNR3= 10 * np.log10(abs(sensor_data_np1[i][10]-column_means[10]) / abs(column_means[10]))
        SNR4= 10 * np.log10(abs(sensor_data_np1[i][13]-column_means[13]) / abs(column_means[13]))
        SNR5= 10 * np.log10(abs(sensor_data_np1[i][16]-column_means[16]) / abs(column_means[16]))
        SNR6= 10 * np.log10(abs(sensor_data_np1[i][19]-column_means[19]) / abs(column_means[19]))
        SNR7= 10 * np.log10(abs(sensor_data_np1[i][22]-column_means[22]) / abs(column_means[22]))
        SNR8= 10 * np.log10(abs(sensor_data_np1[i][25]-column_means[25]) / abs(column_means[25]))
        SNR9= 10 * np.log10(abs(sensor_data_np1[i][28]-column_means[28]) / abs(column_means[28]))
        # SNR1= 10 * np.log10(abs(B[1]*1e6) / abs(B_geo+B_env1+B_noise))
        result.append(SNR)
        result1.append(SNR1)
        result2.append(SNR2)
        result3.append(SNR3)
        result4.append(SNR4)
        result5.append(SNR5)
        result6.append(SNR6)
        result7.append(SNR7)
        result8.append(SNR8)
        result9.append(SNR9)
        if(SNR<SNRstd and snrID==0):
            snrindex = i-1
            snrID=snrID+1
    # result1.append(SNR1)
    # 更新位置，例如每次迭代使磁铁沿 z 轴远离传感器
    # y -= 0.1  # 每次迭代z增加1单位距离

    # 打印当前位置和方向
    # print(f"Iteration {i}: Position ({x}, {y}, {z}), Orientation (theta: {theta}, phi: {phi})")

    # 更新初始位置为下一次
    # init = [x, y, z, theta, phi]

# print(B_res[100])
def find_nearest_std_crossing(distances, snr_values):
    """
    Find the distance where the SNR value is nearest to zero.

    Args:
    distances (list): List of distance values.
    snr_values (list): Corresponding list of SNR values.

    Returns:
    float: The distance where SNR is nearest to zero.
    """
    min_diff = float('inf')
    nearest_distance = None
    for d, snr in zip(distances, snr_values):
        diff = abs(snr)  # Get the absolute difference from zero
        if diff < min_diff:
            min_diff = diff
            nearest_distance = d
    return nearest_distance

def find_nearest_zero_crossing(distances, snr_values):
    """
    Find the distance where the SNR value is nearest to zero.

    Args:
    distances (list): List of distance values.
    snr_values (list): Corresponding list of SNR values.

    Returns:
    float: The distance where SNR is nearest to zero.
    """
    min_diff = float('inf')
    nearest_distance = None
    for d, snr in zip(distances, snr_values):
        diff = abs(snr)  # Get the absolute difference from zero
        if diff < min_diff:
            min_diff = diff
            nearest_distance = d
    return nearest_distance
plt.figure(figsize=(10, 6))
# Find the nearest zero crossing points for both curves
cross_point_5 = find_nearest_zero_crossing(distance, result)
cross_point_20 = distance[disindex]
snrstdmin = distance[snrindex]

# Plot the graph
# plt.plot(distance, result,label="SNR")
# plt.ylabel("SNR (dB)")
# plt.legend(loc='upper left')
# plt.plot(distance, relativepos,label="realtivepos")
# plt.plot(distance, result1, label="B_env = 20")

# Draw a horizontal line at SNR = 0
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(SNRstd, color='c', linestyle='--')
# Mark the nearest zero crossing points on the graph
# if cross_point_5 is not None:
#     plt.axvline(cross_point_5, color='red', linestyle='--')
#     plt.text(cross_point_5, 1, f'{cross_point_5:.2f} cm', color='blue', verticalalignment='bottom')

if cross_point_5 is not None:
    plt.axvline(cross_point_5, color='blue', linestyle='--')
    plt.text(cross_point_5, 1, f'{cross_point_5:.2f} cm', color='blue', verticalalignment='bottom')

if cross_point_20 is not None:
    plt.axvline(cross_point_20, color='green', linestyle='--')
    plt.text(cross_point_20, -5, f'{cross_point_20:.2f} cm', color='green', verticalalignment='bottom')

if snrstdmin is not None:
    plt.axvline(snrstdmin, color='red', linestyle='--')
    plt.text(snrstdmin, -3, f'{snrstdmin:.2f} cm', color='red', verticalalignment='bottom')


# plt.title("SNR over Distance")
# plt.xlabel("Distance (cm)",fontsize="20")
# plt.ylabel("SNR (dB)",fontsize="20")
# plt.tick_params(axis='both', which='major', labelsize=20)
# y1_min, y1_max = plt.gca().get_ylim()

# ax2 = plt.gca().twinx()

# # 绘制 relativepos 数据
# ax2.plot(distance, relativepos, label="Relative Position", color='green')
# ax2.set_ylabel("Relative Position",fontsize="20")
# ax2.set_ylim(y1_min, y1_max)
# # ax2.legend(loc='upper right')
# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.legend()
# plt.grid()
# plt.show()
line3, = plt.plot(distance, result1,label="sensor 5")
line4, = plt.plot(distance, result2,label="sensor 7")
line5, = plt.plot(distance, result3,label="sensor 2")
line6, = plt.plot(distance, result4,label="sensor 11")
line7, = plt.plot(distance, result5,label="sensor 25")
line8, = plt.plot(distance, result6,label="sensor 15")
line9, = plt.plot(distance, result7,label="sensor 16")
line10, = plt.plot(distance, result8,label="sensor 3")
line11, = plt.plot(distance, result9,label="sensor 26")

ax1 = plt.gca()
line1, = ax1.plot(distance, result, label="sensor4", color='blue')

# # 创建共享x轴的次坐标轴
ax2 = ax1.twinx()

line2, = ax2.plot(distance, relativepos, label="Relative Position", color='green')

# 设置轴标签
ax1.set_xlabel("Distance (cm)", fontsize=15)
ax1.set_ylabel("SNR (dB)", fontsize=15)


ax2.set_ylabel("Relative Position (cm)", fontsize=15)

# 创建图例项
lines = [line2,line1, line3,line4,line5,line6,line7,line8,line9,line10,line11]
labels = [l.get_label() for l in lines]

# 显示图例，可以选择放在图表的上方、下方、左侧或右侧
ax1.legend(lines, labels, loc='upper right',fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax2.tick_params(axis='both', which='major', labelsize=15)

# ax2 = plt.gca().twinx()

# 绘制 relativepos 数据
# ax2.plot(distance, relativepos, label="Relative Position", color='green')
# ax2.set_ylabel("Relative Position",fontsize="20")
y1_min, y1_max = ax1.get_ylim()  # 获取 ax1 的 y 轴范围
y2_min, y2_max = ax2.get_ylim()  # 获取 ax2 的 y 轴范围

# 设置统一的 y 轴范围
common_min = min(y1_min, y2_min)
common_max = max(y1_max, y2_max)
# common_min = min(-40, -40)
# common_max = max(y1_max, y2_max)
ax1.set_ylim(common_min, common_max)
ax2.set_ylim(common_min, common_max)
# ax2.legend(loc='upper right')
# plt.legend()
plt.grid()
plt.show()