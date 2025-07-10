import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
# 读取 CSV 文件
file_path = "/home/czy/桌面/mapdata/map4.csv"  # 请替换为你的文件路径
# file_path = "/home/czy/桌面/magx-main1/magneticdata.csv"  # 请替换为你的文件路径
df = pd.read_csv(file_path)

# 假设列名依次为 "timestamp", "x", "y", "z", "latitude", "longitude"
df.columns = ["timestamp", "x", "y", "z", "latitude", "longitude"]

# 将时间戳转换为日期时间格式（假设时间戳是毫秒级的）
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# 如果时间是 UTC，需要转换为北京时间
df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=8)  # 转换为 UTC+8 (北京时间)

# 绘制 x、y、z 随时间变化的曲线
plt.figure(figsize=(12, 6))

plt.plot(df["timestamp"], df["x"], label="X", color="b", alpha=0.7)
plt.plot(df["timestamp"], df["y"], label="Y", color="r", alpha=0.7)
plt.plot(df["timestamp"], df["z"], label="Z", color="g", alpha=0.7)

print("x:mean",np.mean(df["x"]))
print("x:std",np.std(df["x"]))
print("y:mean",np.mean(df["y"]))
print("y:std",np.std(df["y"]))
print("z:mean",np.mean(df["z"]))
print("z:std",np.std(df["z"]))


plt.xlabel("Time", fontsize=22)  # 修改 X 轴标签字体大小
plt.ylabel("Position (cm)", fontsize=22)  # 修改 Y 轴标签字体大小
# plt.title("X, Y, and Z over Time", fontsize=25)  # 修改标题字体大小
# plt.legend(fontsize=20)  # 修改图例字体大小
plt.legend(loc="upper center", fontsize=18, bbox_to_anchor=(0.5, 1.15), ncol=4)
# 使用 DateFormatter 设置横坐标格式为时:分
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=30, fontsize=18)  # 旋转 X 轴刻度并修改字体大小
# plt.xticks(fontsize=20)  # 旋转 X 轴刻度并修改字体大小
plt.tight_layout()

plt.yticks(fontsize=18)  # 修改 Y 轴刻度字体大小
plt.grid(True)
plt.savefig("Figure19_j.jpg",dpi=300)
plt.show()
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 读取 CSV 文件
# file_path = "/home/czy/Downloads/data.csv"  # 请替换为你的文件路径
# df = pd.read_csv(file_path)

# # 假设列名依次为 "timestamp", "x", "y", "z", "latitude", "longitude"
# df.columns = ["timestamp", "x", "y", "z", "latitude", "longitude"]

# # 将时间戳转换为日期时间格式（假设时间戳是毫秒级的）
# df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# # 如果时间是 UTC，需要转换为北京时间
# df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=8)  # 转换为 UTC+8 (北京时间)

# # 筛选出晚上19:00之后的数据
# df_filtered = df[df["timestamp"].dt.hour >= 19]

# # 创建 3D 图
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # 绘制经纬度与 y 的关系
# ax.scatter(df_filtered["longitude"], df_filtered["latitude"], df_filtered["y"], c=df_filtered["y"], cmap='viridis')

# # 设置标签
# ax.set_xlabel('Longitude', fontsize=15)
# ax.set_ylabel('Latitude', fontsize=15)
# ax.set_zlabel('Y Value', fontsize=15)
# ax.set_title('Latitude, Longitude vs Y Value (After 19:00)', fontsize=18)

# # 显示图形
# plt.show()
