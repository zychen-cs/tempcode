import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
result = []
result1 = []
distance=[]
B_res=[]
import re
position = pd.read_csv("/home/czy/桌面/MetaJoint-master/Code/tracking_method/data/24_06_16/original_data/result_joint2_0616_1_1.csv")
# position = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0114test2_1.csv")
# [0, -0.8, 0],
# [ 0.5, -0.5, 0],
# [-0.5, -0.5, 0],
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
csv_file_path = "/home/czy/桌面/MetaJoint-master/Code/tracking_method/data/24_06_16/calibration_joint2_0616_1.csv"
# csv_file_path = '/home/czy/桌面/magx-main1/0114calib_3.csv'
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
column_means = np.min(sensor_data_np, axis=0)
column_stds = np.std(sensor_data_np, axis=0)
print(column_means[1])
print(column_stds)
csv_file_path1 = '/home/czy/桌面/MetaJoint-master/Code/tracking_method/data/24_06_16/original_data/readings_joint2_0616_1_1.csv'
# csv_file_path1 = '/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0114sensor2_1.csv'
sensor_data_df1 = pd.read_csv(csv_file_path1)

# Apply the process_row function to each row of sensor data
# Excluding the first two columns which are 'Unnamed: 0' and 'Time Stamp'
sensor_columns1 = sensor_data_df1.columns[2:]
sensor_data_processed1 = sensor_data_df1[sensor_columns1].apply(process_row, axis=1)

# Convert the series of lists into a 2D numpy array
sensor_data_np1 = np.array(sensor_data_processed1.tolist())
# position = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0114test2_1.csv")
position = pd.read_csv("/home/czy/桌面/MetaJoint-master/Code/tracking_method/data/24_06_16/original_data/result_joint2_0616_1_1.csv")
result=[]
result_sensor=[]
relativepos=[]
relativepos.append(0)
distance=[]
SNRstd= 10 * np.log10(column_stds[1] / abs(column_means[1]))
disID=0
snrID=0
snrID1=0
disindex=0
snrindex=0

snrindex1=0
for i in range(0,len(sensor_data_np1)):
    if (i>0):
        relativepos.append(np.sqrt(position["x"][i]**2+position["y"][i]**2+position["z"][i]**2)-np.sqrt(position["x"][i-1]**2+position["y"][i-1]**2+position["z"][i-1]**2))
        if(relativepos[i]>1 and disID==0):
            disID=disID+1
            disindex = i-1
    # tempdata = sensor_data_np1[i][1] * (1 + coeff_sen[i, 0] * (Temper - 35) * factorSen)
    if(column_means[1]>0):
        distance.append(np.sqrt(position["x"][i]**2+position["y"][i]**2+position["z"][i]**2))
        # print(abs(sensor_data_np1[i][1]-column_means[1]))
        SNR= 10 * np.log10(abs(sensor_data_np1[i][1]-column_means[1]) / abs(column_means[1]))
        # SNR1= 10 * np.log10(abs(B[1]*1e6) / abs(B_geo+B_env1+B_noise))
        result_sensor.append(SNR)
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

for i in range(0,len(position)):
    
    # distance.append(np.sqrt(position["x"][i]**2+position["y"][i]**2+position["z"][i]**2))
    # x, y, z, theta, phi = init
    x = position["x"][i]
    y = position["y"][i]
    z = position["z"][i]
    theta = position["theta"][i]
    phi = position["phy"][i]

    Xs = 0
    Ys = -0.8*1e-2
    Zs = 0

    # B_geo = 40
    # B_env = 5
    # B_env1 = 30
    B_noise = column_means[1]
    
    VecM = np.array([np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)]) * 1e-7 * 0.25
    VecR = np.array([Xs - x*(1e-2), Ys - y*(1e-2), Zs - z*(1e-2)])
    
# 计算 VecR 的范数
    NormR = np.linalg.norm(VecR)

        # print(VecM)
        # print(VecR)
    # 计算矢量 B
        # B = (3.0 * VecR * np.dot(VecM.T, VecR) / NormR ** 5) - VecM / NormR ** 3
    scalar_part = 3.0 * (np.dot(VecM, VecR) / NormR**5)
    B = scalar_part * VecR - VecM / NormR**3
    B_res.append(B)
    # print(B[1]*1e6)
    SNR= 10 * np.log10(abs(B[1]*1e6) / abs((B_noise)))
    if(SNR<SNRstd and snrID1==0):
            snrindex1 = i-1
            snrID1=snrID1+1
    # SNR1= 10 * np.log10(abs(B[1]*1e6) / abs(B_geo+B_env1+B_noise))
    result.append(SNR)
    # result1.append(SNR1)
    # 更新位置，例如每次迭代使磁铁沿 z 轴远离传感器
    # y -= 0.1  # 每次迭代z增加1单位距离

    # 打印当前位置和方向
    # print(f"Iteration {i}: Position ({x}, {y}, {z}), Orientation (theta: {theta}, phi: {phi})")

    # 更新初始位置为下一次
    # init = [x, y, z, theta, phi]

# print(B_res[100])
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
snrstdmin = distance[snrindex]
snrstdmin1 = distance[snrindex1]
# Find the nearest zero crossing points for both curves
cross_point_5 = find_nearest_zero_crossing(distance, result)
cross_point_20 = find_nearest_zero_crossing(distance, result_sensor)

# Plot the graph
plt.plot(distance, result,label="dipole model")
plt.plot(distance, result_sensor,label = "sensor")
# plt.plot(distance, result1, label="B_env = 20")

# Draw a horizontal line at SNR = 0
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(SNRstd, color='black', linestyle='--')
# Mark the nearest zero crossing points on the graph
if cross_point_5 is not None:
    plt.axvline(cross_point_5, color='blue', linestyle='--')
    plt.text(cross_point_5, 1, f'{cross_point_5:.2f} cm', color='blue', verticalalignment='bottom')

if cross_point_20 is not None:
    plt.axvline(cross_point_20, color='green', linestyle='--')
    plt.text(cross_point_20, -5, f'{cross_point_20:.2f} cm', color='green', verticalalignment='bottom')
if snrstdmin is not None:
    plt.axvline(snrstdmin, color='red', linestyle='--')
    plt.text(snrstdmin, -3, f'{snrstdmin:.2f} cm', color='red', verticalalignment='bottom')
if snrstdmin1 is not None:
    plt.axvline(snrstdmin1, color='c', linestyle='--')
    plt.text(snrstdmin1, -3, f'{snrstdmin1:.2f} cm', color='c', verticalalignment='bottom')

plt.xlabel("Distance (cm)",fontsize=15)
plt.ylabel("SNR (dB)",fontsize=15)
plt.legend(fontsize=15)
plt.tick_params(labelsize=15)
plt.grid()
plt.show()
   



