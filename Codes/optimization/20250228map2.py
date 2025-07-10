import folium
import pandas as pd
from geopy.distance import geodesic
import math

# 读取CSV文件
df = pd.read_csv('/home/czy/桌面/mapdata/map5.csv')
df = pd.read_csv('/home/czy/桌面/magx-main1/magnetic_cordinate_filter.csv')
# df.columns = ["timestamp", "x", "y", "z", "纬度", "经度"]
# df.columns = ["timestamp", "x", "y", "z", "纬度", "经度"]
# 创建地图（使用模糊背景）
map_center = [df['纬度'][0], df['经度'][0]]
# mymap = folium.Map(location=map_center, zoom_start=15, tiles=None)
mymap = folium.Map(location=map_center, zoom_start=15, tiles="CartoDB PositronNoLabels")
# 添加模糊化地图背景（降低透明度）
# folium.TileLayer('CartoDB Positron', opacity=0.3).add_to(mymap)
folium.TileLayer('CartoDB Positron', opacity=0.1).add_to(mymap)  # 降低透明度

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

# 筛选出不重复的经纬度
unique_coordinates = []
for lat, lon in coordinates:
    current_point = (lat, lon)
    if not unique_coordinates or unique_coordinates[-1] != current_point:
        unique_coordinates.append(current_point)

# 重新处理过滤后的不重复经纬度
for lat, lon in unique_coordinates:
    current_point = (lat, lon)
    if prev_point:
        distance = geodesic(prev_point, current_point).meters
        if distance > distance_threshold:
            if segment:
                filtered_segments.append(segment)
            segment = []  # 开始新的轨迹段
    segment.append(current_point)
    prev_point = current_point

if segment:
    filtered_segments.append(segment)

# 在地图上绘制分段轨迹
for segment in filtered_segments:
    folium.PolyLine(segment, color="blue", weight=2.5, opacity=1).add_to(fg)

# 添加标记
for lat, lon in unique_coordinates:
    folium.Marker([lat, lon]).add_to(mymap)

# 上海交大闵行校区的纬度
latitude_sjtu = 31.03  

# 计算 100m 在纬度和经度上的变化
lat_scale = 100 / 111320  # 100m 对应纬度变化
lon_scale = 100 / (111320 * math.cos(math.radians(latitude_sjtu)))  # 100m 对应经度变化

# # **向左下方移动比例尺**
# scale_lat = map_center[0] - 5 * lat_scale  # 让比例尺进一步向下
# scale_lon_start = map_center[1] + 6 * lon_scale  # 让比例尺进一步向左
# scale_lon_end = scale_lon_start - lon_scale  # 保持比例尺长度

# # 绘制比例尺
# folium.PolyLine([(scale_lat, scale_lon_start), (scale_lat, scale_lon_end)], 
#                 color="red", weight=3).add_to(mymap)

# # 添加比例尺标注
# folium.Marker(
#     (scale_lat, scale_lon_end),
#     icon=folium.DivIcon(
#         html="""
#         <div style="
#             font-size: 14px;
#             font-weight: bold;
#             color: white;
#             background-color: rgba(0, 0, 0, 0.7);
#             padding: 4px 8px;
#             border-radius: 5px;
#             text-align: center;
#             box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
#         ">
#             100m
#         </div>
#         """
#     ),
# ).add_to(mymap)

# **计算比例尺位置**
scale_lat = map_center[0] - 7 * lat_scale  # 适当向下移动
scale_lon_start = map_center[1] + 4* lon_scale  # 适当向左移动
scale_lon_end = scale_lon_start - lon_scale  # 100m 长度

# **绘制主比例线**
folium.PolyLine([(scale_lat, scale_lon_start), (scale_lat, scale_lon_end)], 
                color="black", weight=3).add_to(mymap)

# **绘制两端突出的小线段**
offset = 0.0001 # 端点向上偏移量 (单位: 纬度)

folium.PolyLine([(scale_lat, scale_lon_start), (scale_lat + offset, scale_lon_start)], 
                color="black", weight=3).add_to(mymap)

folium.PolyLine([(scale_lat, scale_lon_end), (scale_lat + offset, scale_lon_end)], 
                color="black", weight=3).add_to(mymap)

# **添加比例尺标注**
# folium.Marker(
#     (scale_lat - 2 * offset, (scale_lon_start + scale_lon_end) / 2),  # 文字居中
#     icon=folium.DivIcon(
#         html="""
#         <div style="
#             font-size: 14px;
#             font-weight: bold;
#             color: black;
#             padding: 2px 6px;
#             text-align: center;
#         ">
#             100m
#         </div>
#         """
#     ),
# ).add_to(mymap)

# **添加比例尺标注到右侧外侧**
folium.Marker(
    (scale_lat+0.0001, scale_lon_end + 0.0015),  # 右侧外移一点
    icon=folium.DivIcon(
        html="""
        <div style="
            font-size: 14px;
            font-weight: bold;
            color: black;
            text-align: left;
            white-space: nowrap;
        ">
            100m
        </div>
        """
    ),
).add_to(mymap)



# folium.Marker((scale_lat, scale_lon_end), 
#               icon=folium.DivIcon(html="<div style='color: black; font-size: 15px;'>100m</div>")).add_to(mymap)

# 添加图层控制
folium.LayerControl().add_to(mymap)

# 保存地图
mymap.save("newmap.html")

print("地图已生成，保存为 maptest.html")
