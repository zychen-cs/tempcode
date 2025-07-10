import folium
import pandas as pd
from geopy.distance import geodesic

# 读取CSV文件
df = pd.read_csv('/home/czy/桌面/magx-main1/magnetic_cordinate_filter.csv')

# 创建地图（去除地理信息名称）
map_center = [df['纬度'][0], df['经度'][0]]
mymap = folium.Map(location=map_center, zoom_start=15, tiles="CartoDB PositronNoLabels")

# 轨迹图层
fg = folium.FeatureGroup(name="轨迹线").add_to(mymap)

# 提取经纬度坐标
coordinates = list(zip(df['纬度'], df['经度']))

# 设置距离阈值 (单位: 米)
distance_threshold = 50  

# 处理GPS丢失，避免错误连接
filtered_segments = []
segment = []
prev_point = None

for lat, lon in coordinates:
    current_point = (lat, lon)
    if prev_point:
        distance = geodesic(prev_point, current_point).meters
        if distance > distance_threshold:
            # 断开轨迹，将当前段添加到列表
            if segment:
                filtered_segments.append(segment)
            segment = []  # 开始新的轨迹段
    segment.append(current_point)
    prev_point = current_point

# 添加最后的轨迹段
if segment:
    filtered_segments.append(segment)

# 在地图上绘制分段轨迹
for segment in filtered_segments:
    folium.PolyLine(segment, color="blue", weight=2.5, opacity=1).add_to(fg)

# 添加标记
for lat, lon in coordinates:
    folium.Marker([lat, lon]).add_to(mymap)

# 添加图层控制
folium.LayerControl().add_to(mymap)

# 保存地图
mymap.save("trajectory_map.html")

print("地图已生成，保存为 trajectory_map.html")
