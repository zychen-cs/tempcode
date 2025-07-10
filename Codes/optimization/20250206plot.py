import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
before = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0206MLP_before.csv")
after = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0206MLP_after.csv")

# 提取 x, y, z 列
timesteps = range(1,len(before)+1)  # 假设数据点数量相同

plt.figure(figsize=(12, 6))

# 定义颜色
colors = {'x': 'r', 'y': 'g', 'z': 'b'}  # 例如：红、绿、蓝

for axis in ['x', 'y', 'z']:
    plt.plot(timesteps, before[axis], label=f'Before {axis}', linestyle='--', color=colors[axis])
    plt.plot(timesteps, after[axis], label=f'After {axis}', linestyle='-', color=colors[axis])

plt.xlabel("Index", fontsize=22)
plt.ylabel("Position(cm)", fontsize=22)
plt.xticks(ticks=range(1, 16), labels=range(1, 16), fontsize=22)

plt.yticks(fontsize=22)
plt.legend(fontsize=15)
plt.grid(axis='y')

plt.savefig("Figure8_b.jpg",dpi=300)
# plt.title("Comparison of x, y, z Before and After Movement", fontsize=18)

plt.show()
