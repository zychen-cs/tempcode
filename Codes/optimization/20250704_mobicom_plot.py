import pandas as pd
import numpy as np

# ---------- Step 1: 读取数据 ----------
data = pd.read_csv("/media/czy/T7 Shield/mobicom_dataset/knownoise_LM/0704LM_mag2_8_1.csv")
# data = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0707LM_mag88_pos6_1.csv")
# ---------- Step 2: 提取预测值 ----------
x_pred = data["x"].mean()
y_pred = data["y"].mean()
z_pred = data["z"].mean()

theta_pred = data["theta"].values
phi_pred = data["phy"].values

# ---------- Step 3: Ground truth ----------
gt_position = np.array([2, -7, 0.7])
gt_position = np.array([0, -7, 0.7])
gt_position = np.array([-2, -7, 0.7])
gt_position = np.array([2, -5, 0.7])
# gt_position = np.array([0, -5, 0.7])
# gt_position = np.array([-2, -5, 0.7])
gt_orientation = np.array([0.0, 0.0])  # theta, phi

# ---------- Step 4: 计算 Position Error ----------
pred_position = np.array([x_pred, y_pred, z_pred])
pos_error = np.linalg.norm(pred_position - gt_position)
print(f"Mean Position Error (Euclidean Distance): {pos_error:.4f}")

# ---------- Step 5: 计算 Orientation Error ----------
def spherical_to_unit_vec(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)

# 真实方向单位向量（广播成同样长度）
theta_gt = np.full_like(theta_pred, gt_orientation[0])
phi_gt = np.full_like(phi_pred, gt_orientation[1])

vec_gt = spherical_to_unit_vec(theta_gt, phi_gt)
vec_pred = spherical_to_unit_vec(theta_pred, phi_pred)

dot_product = np.sum(vec_gt * vec_pred, axis=1)
dot_product = np.clip(dot_product, -1.0, 1.0)
angle_diff_rad = np.arccos(dot_product)
angle_diff_deg = np.degrees(angle_diff_rad)

mean_orientation_error = np.mean(angle_diff_deg)
print(f"Mean Orientation Error (Angular Difference in Degrees): {mean_orientation_error:.4f}")
