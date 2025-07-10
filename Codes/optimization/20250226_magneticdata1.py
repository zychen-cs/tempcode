import pandas as pd

# 读取 CSV 文件
file_path = "/home/czy/桌面/mapdata/map5.csv"  # 请替换为你的文件路径
df1 = pd.read_csv(file_path)
df1.columns = ["timestamp", "x", "y", "z", "latitude", "longitude"]
df = pd.read_csv(file_path)

# 假设列名依次为 "timestamp", "x", "y", "z", "latitude", "longitude"
df.columns = ["timestamp", "x", "y", "z", "latitude", "longitude"]

# 将时间戳转换为日期时间格式（假设时间戳是毫秒级的）
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# 如果时间是 UTC，需要转换为北京时间
df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=8)  # 转换为 UTC+8 (北京时间)

# 计算时间戳的差异（单位: 秒）
df["time_diff"] = df["timestamp"].diff().dt.total_seconds()

# 筛选出时间差大于 10 秒的位置
threshold = 10  # 时间戳突变阈值为 10 秒
time_diff_changes = df[df["time_diff"] > threshold]

# 打印这些突变点的时间戳
print("突变的时间戳（前后时间差大于 10 秒）:")
# print(df1["timestamp"])
print(time_diff_changes["timestamp"])
