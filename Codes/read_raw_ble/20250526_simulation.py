import numpy as np

# 传感器阵列位置 (单位：米)
sensors = 1e-2 * np.array([
    [0.5, -0.5, 0],
    [-1, 0, 0],
    [-0.5, -0.5, 0],
    [0, -0.8, 0],
    [1, 0, 0],
    [0.5, 0.5, 0],
    [0, 0, 0],
    [-0.5, 0.5, 0],
])

psensor_circular = 1e-2 * np.array([
    [-0.6, 0, 0],
    [0.6, 0, 0],
    [0, -0.5, 0],
    [0.9, 0.5, 0],
    [0.3, 0.5, 0],
    [0, 0, 0],
    [-0.3, 0.5, 0],
    [-0.9, 0.5, 0]
])

# 磁铁位置（单位米）和姿态（theta, phi）
magnet_pos = np.array([2.5, 4.5, 2]) * 1e-2  # 注意单位统一为米
theta = np.pi
phi = 0

# 磁矩向量 (单位A·m^2)，假设磁矩大小m=0.46，方向由theta, phi决定
magnitude_m = 0.46
VecM = np.array([
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta)
]) * 1e-7 * np.exp(np.log(magnitude_m))  # 假设 m 值为 0.46

def dipole_B_field(magnet_pos, VecM, sensor_pos):
    """
    计算磁偶极子在sensor_pos处的磁场矢量 (单位: Tesla)
    """
    VecR = sensor_pos - magnet_pos
    NormR = np.linalg.norm(VecR)
    if NormR < 1e-9:
        return np.zeros(3)  # 避免除零
    B = (3.0 * VecR * (np.dot(VecM, VecR)) / NormR**5 - VecM / NormR**3)
    return B

# 计算 layout A dipole 模拟磁场 (假设代表真实磁场，实际中应替换为真实测量数据)
B1_dipole = np.array([dipole_B_field(magnet_pos, VecM, sensor) for sensor in sensors])

# 假设真实磁场 B1_real = B1_dipole + 一些小噪声，模拟真实数据更接近真实
noise = np.random.normal(scale=0.05 * np.max(np.linalg.norm(B1_dipole, axis=1)), size=B1_dipole.shape)
B1_real = B1_dipole + noise

# 计算 layout B dipole 模拟磁场
B2_dipole = np.array([dipole_B_field(magnet_pos, VecM, sensor) for sensor in psensor_circular])

# 计算幅值比，避免除0，增加小常数平滑
norm_B1 = np.linalg.norm(B1_dipole, axis=1) + 1e-15
norm_B2 = np.linalg.norm(B2_dipole, axis=1) + 1e-15
scale = norm_B2 / norm_B1  # (8,)

# 将 layout A 真实磁场按照幅值比缩放，得到 layout B 估计磁场
B2_est = B1_real * scale[:, np.newaxis]  # 广播乘法

# 打印对比
print("Layout B dipole 计算磁场 (uT):")
print(B2_dipole * 1e6)

print("\nLayout B 估计磁场 (通过 layout A 实际磁场和幅值比放缩) (uT):")
print(B2_est * 1e6)

# 计算估计误差（幅值误差）
error_norm = np.linalg.norm(B2_est* 1e6 - B2_dipole* 1e6, axis=1)
print("\n估计误差（幅值差，单位Tesla）:")
print(error_norm)
