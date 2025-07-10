import matplotlib.pyplot as plt
import numpy as np

result = []
result1 = []
distance=[]
B_res=[]
# [0, -0.8, 0],
# [ 0.5, -0.5, 0],
# [-0.5, -0.5, 0],
# x = -0.62
# y = -6.16
# z = 1.99
init = [-0.62, -6.16, 1.99, 0, 0]  # 初始位置 (x, y, z) 和方向 (theta, phi)
for i in range(0,1000):
    distance.append(-(-(i)*0.1-3))
    x, y, z, theta, phi = init
    Xs = 0
    Ys = -0.8*1e-2
    Zs = 0
    B_geo = 40
    B_env = 10
    B_env1 = 20
    B_noise = 0.5
    
    VecM = np.array([np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)]) * 1e-7 * np.exp(0.25)
    VecR = np.array([Xs - (x*1e-2), Ys - (y*1e-2), Zs - (z*1e-2)])
    
# 计算 VecR 的范数
    NormR = np.linalg.norm(VecR)

        # print(VecM)
        # print(VecR)
    # 计算矢量 B
        # B = (3.0 * VecR * np.dot(VecM.T, VecR) / NormR ** 5) - VecM / NormR ** 3
    scalar_part = 3.0 * (np.dot(VecM, VecR) / NormR**5)
    B = scalar_part * VecR - VecM / NormR**3
    B_res.append(B)
    print(B*1e6)
    SNR= 10 * np.log10(abs(B[1]*1e6) / abs((B_geo+B_env+B_noise)))
    SNR1= 10 * np.log10(abs(B[1]*1e6) / abs(B_geo+B_env1+B_noise))
    result.append(SNR)
    result1.append(SNR1)
    # 更新位置，例如每次迭代使磁铁沿 z 轴远离传感器
  

    # 打印当前位置和方向
    # print(f"Iteration {i}: Position ({x}, {y}, {z}), Orientation (theta: {theta}, phi: {phi})")

    # 更新初始位置为下一次
    init = [x, y, z, theta, phi]

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
# print(distance)
# Find the nearest zero crossing points for both curves
cross_point_5 = find_nearest_zero_crossing(distance, result)
cross_point_20 = find_nearest_zero_crossing(distance, result1)

# Plot the graph
plt.plot(distance, result, label="B_env = 10")
plt.plot(distance, result1, label="B_env = 20")

# Draw a horizontal line at SNR = 0
plt.axhline(0, color='gray', linestyle='--')

# Mark the nearest zero crossing points on the graph
if cross_point_5 is not None:
    plt.axvline(cross_point_5, color='blue', linestyle='--')
    plt.text(cross_point_5, 1, f'{cross_point_5:.2f} cm', color='blue', verticalalignment='bottom')

if cross_point_20 is not None:
    plt.axvline(cross_point_20, color='green', linestyle='--')
    plt.text(cross_point_20, -5, f'{cross_point_20:.2f} cm', color='green', verticalalignment='bottom')


# plt.title("SNR over Distance")
plt.xlabel("Distance (cm)",fontsize=15)
plt.ylabel("SNR (dB)",fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=15)
plt.grid()
plt.show()
   



