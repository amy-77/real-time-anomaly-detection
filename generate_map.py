import folium
import math

# 站点数据 (从数据库查询结果填入)
stations = [
    {"id": "amfissa", "name": "Amfissa", "lat": 38.52491, "lon": 22.386218, "elev": 168.0},
    {"id": "dodoni", "name": "Dodoni", "lat": 39.556817, "lon": 20.78555, "elev": 675.0},
    {"id": "embonas", "name": "Embonas", "lat": 36.224385, "lon": 27.855402, "elev": 430.0},
    {"id": "grevena", "name": "Grevena", "lat": 40.08919, "lon": 21.445693, "elev": 510.0},
    {"id": "heraclion", "name": "Heraclion", "lat": 35.300751, "lon": 25.163781, "elev": 115.0},
    {"id": "kolympari", "name": "Kolympari", "lat": 35.52483, "lon": 23.79876, "elev": 40.0},
    {"id": "makrinitsa", "name": "Makrinitsa", "lat": 39.405349, "lon": 22.987778, "elev": 850.0},
    {"id": "portaria", "name": "Portaria", "lat": 39.38786976, "lon": 22.99513725, "elev": 600.0},
    {"id": "sparti", "name": "Sparti", "lat": 37.053581, "lon": 22.437633, "elev": 204.0},
    {"id": "uth_volos", "name": "Volos - Uth", "lat": 39.36076, "lon": 22.93165, "elev": 9.0},
    {"id": "vlasti", "name": "Vlasti", "lat": 40.4584, "lon": 21.519036, "elev": 1194.0},
    {"id": "volos", "name": "Volos", "lat": 39.3744678, "lon": 22.9619388, "elev": 52.0},
    {"id": "volos-port", "name": "Volos - Port", "lat": 39.357552, "lon": 22.950442, "elev": 20.0},
    {"id": "zagora", "name": "Zagora", "lat": 39.4482, "lon": 23.100689, "elev": 505.0}
]

# Haversine 距离计算
def get_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# 配置参数
MAX_DIST_KM = 50      # 距离阈值
MAX_ELEV_DIFF_M = 500 # 海拔差阈值

# 创建地图 (中心点设在希腊中部)
m = folium.Map(location=[38.5, 24.0], zoom_start=7, tiles="CartoDB positron")

# 1. 绘制所有站点
for s in stations:
    tooltip = f"<b>{s['name']}</b><br>ID: {s['id']}<br>Elev: {s['elev']}m"
    folium.CircleMarker(
        location=[s['lat'], s['lon']],
        radius=6,
        popup=tooltip,
        tooltip=s['name'],
        color="#3388ff",
        fill=True,
        fill_color="#3388ff"
    ).add_to(m)

# 2. 绘制邻居连线
processed_pairs = set()
links_count = 0

for i, s1 in enumerate(stations):
    for j, s2 in enumerate(stations):
        if i >= j: continue # 避免重复
        
        dist = get_distance(s1['lat'], s1['lon'], s2['lat'], s2['lon'])
        elev_diff = abs(s1['elev'] - s2['elev'])
        
        # 判断是否为邻居
        if dist <= MAX_DIST_KM and elev_diff <= MAX_ELEV_DIFF_M:
            # 画线
            folium.PolyLine(
                locations=[[s1['lat'], s1['lon']], [s2['lat'], s2['lon']]],
                color="red",
                weight=1.5,
                opacity=0.6,
                tooltip=f"Dist: {dist:.1f}km, ElevDiff: {elev_diff:.0f}m"
            ).add_to(m)
            links_count += 1

# 保存
outfile = "spatial_network_map.html"
m.save(outfile)
print(f"✅ 地图已生成: {outfile}")
print(f"🔗 总连接数: {links_count} (阈值: <{MAX_DIST_KM}km, <{MAX_ELEV_DIFF_M}m)")

