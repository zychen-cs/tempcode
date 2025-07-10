import matplotlib.pyplot as plt
import numpy as np
x_B=[]
y_B=[]
z_B=[]
result = []
result1 = []
distance=[]
B_res=[]
# [0, -0.8, 0],
# [ 0.5, -0.5, 0],
# [-0.5, -0.5, 0],
init = [1, -3, 0.5, 0, 0]  # 初始位置 (x, y, z) 和方向 (theta, phi)
for i in range(0,1000):
    distance.append(-(-(i)*0.1-3))
    x, y, z, theta, phi = init
    Xs = 0
    Ys = 0
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
    x_B.append(B[0]*1e6)
    y_B.append(B[1]*1e6)
    z_B.append(B[2]*1e6)

    y -= 0.1  # 每次迭代z增加1单位距离

  
    init = [x, y, z, theta, phi]


plt.figure(figsize=(10, 6))



plt.plot(distance, x_B, label="x")
plt.plot(distance, y_B, label="y")
plt.plot(distance, z_B, label="z")

plt.xlabel("Distance (cm)",fontsize=15)
plt.ylabel("B",fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=15)
plt.grid()
plt.show()
   



