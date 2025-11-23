import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 配置
DB_PATH = '/data/qwang/q/datagem/stream_detection/weather_stream.db'
TARGET_STATION = 'volos'
VARIABLE = 'out_hum'  # 改为湿度
START_TIME = '2025-11-22 11:00:00'
END_TIME = '2025-11-22 17:00:00'

def get_data(station_id):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT time, {VARIABLE} 
        FROM observations 
        WHERE station_id = ? AND time BETWEEN ? AND ?
        ORDER BY time
    """
    df = pd.read_sql_query(query, conn, params=(station_id, START_TIME, END_TIME))
    conn.close()
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def find_real_neighbors(target_id, limit=5):
    conn = sqlite3.connect(DB_PATH)
    stations = pd.read_sql_query("SELECT station_id, latitude, longitude FROM stations", conn)
    conn.close()
    
    target = stations[stations['station_id'] == target_id]
    if target.empty:
        print(f"Target station {target_id} not found!")
        return []
        
    t_lat, t_lon = target.iloc[0]['latitude'], target.iloc[0]['longitude']
    
    distances = []
    for _, row in stations.iterrows():
        if row['station_id'] == target_id:
            continue
        dist = haversine_distance(t_lat, t_lon, row['latitude'], row['longitude'])
        distances.append((row['station_id'], dist))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    
    print(f"\n找到 {target_id} 的最近邻居:")
    neighbors = []
    for sid, dist in distances[:limit]:
        print(f"  - {sid}: {dist:.1f} km")
        if dist < 100: # 只取150km以内的
            neighbors.append(sid)
            
    return neighbors

def main():
    # 1. 找邻居
    neighbors = find_real_neighbors(TARGET_STATION)
    if not neighbors:
        print("没有找到足够近的邻居！")
        return

    # 2. 获取数据
    target_df = get_data(TARGET_STATION)
    neighbor_dfs = {nid: get_data(nid) for nid in neighbors}
    
    # 3. 合并
    if target_df.empty:
        print("目标站点无数据")
        return
        
    target_df.set_index('time', inplace=True)
    target_df.rename(columns={VARIABLE: f'{TARGET_STATION}'}, inplace=True)
    
    combined_df = target_df
    for nid, df in neighbor_dfs.items():
        if not df.empty:
            df.set_index('time', inplace=True)
            df.rename(columns={VARIABLE: nid}, inplace=True)
            combined_df = combined_df.join(df, how='outer')
            
    # 4. 打印
    print(f"\n{'='*100}")
    print(f"数据对比: {TARGET_STATION} vs Neighbors (变量: {VARIABLE})")
    print(f"{'='*100}")
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    
    # 标记异常点
    highlight_times = ['2025-11-22 16:50:00', '2025-11-22 17:00:00']
    
    # Header
    headers = [TARGET_STATION] + neighbors
    print(f"{'Time':20} | " + " | ".join([f"{h:^12}" for h in headers]))
    print("-" * (24 + 15 * len(headers)))
    
    for time_idx, row in combined_df.iterrows():
        time_str = str(time_idx)
        mark = " <--- ⚠️  Anomaly" if time_str in highlight_times else ""
        
        vals = []
        for h in headers:
            val = row.get(h, float('nan'))
            vals.append(f"{val:.0f}%" if pd.notna(val) else "   NaN") # 湿度打印整数更清晰
            
        print(f"{time_str} | " + " | ".join([f"{v:^12}" for v in vals]) + mark)

if __name__ == '__main__':
    main()
