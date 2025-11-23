#!/usr/bin/env python3
"""
Weather Data Anomaly Detection System
-------------------------------------
A comprehensive system for detecting anomalies in weather station data using:
1. Temporal Analysis (ARIMA, STL, Statistical methods)
2. Spatial Verification (Neighbor trend correlation)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import argparse
import json
import warnings
warnings.filterwarnings('ignore')


class WindowDataLoader:
    """Loads data within a sliding window from SQLite."""
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
    
    def get_window_data(self, station_id: str, start_time: str = None, 
                       end_time: str = None, window_hours: int = None) -> pd.DataFrame:
        """
        获取指定站点的滑动窗口数据
        支持两种模式:
        1. 指定 start_time + end_time: 使用精确时间范围
        2. 指定 end_time + window_hours: 从end_time往前推window_hours小时
        """
        # 模式1: 指定了起始和结束时间
        if start_time and end_time:
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
        
        # 模式2: 指定结束时间 + 窗口长度
        elif end_time and window_hours:
            end_dt = pd.to_datetime(end_time)
            start_dt = end_dt - timedelta(hours=window_hours)
        
        else:
            raise ValueError("必须指定: (start_time + end_time) 或 (end_time + window_hours)")
        
        # 查询数据
        query = """
            SELECT time, temp_out, out_hum, wind_speed, bar, rain
            FROM observations
            WHERE station_id = ? AND time BETWEEN ? AND ?
            ORDER BY time ASC
        """
        df = pd.read_sql_query(query, self.conn, 
                               params=(station_id, start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                                     end_dt.strftime('%Y-%m-%d %H:%M:%S')))
        df['time'] = pd.to_datetime(df['time'])
        return df
    
    def get_all_stations(self) -> pd.DataFrame:
        """获取所有站点信息"""
        return pd.read_sql_query("SELECT station_id, station_name_en, latitude, longitude, elevation FROM stations", self.conn)
    
    def close(self):
        if self.conn:
            self.conn.close()




class StatisticalDetector:
    """统计异常检测器"""
    
    @staticmethod
    def detect_3sigma(values: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, Dict]:
        """3σ规则异常检测"""
        if len(values) < 3:
            return np.zeros(len(values), dtype=bool), {}
        
        mean, std = np.mean(values), np.std(values)
        
        if std == 0:  # 数据无变化
            return np.zeros(len(values), dtype=bool), {
                'mean': mean, 'std': 0, 'upper_bound': mean, 
                'lower_bound': mean, 'is_constant': True
            }
        
        upper, lower = mean + threshold * std, mean - threshold * std
        anomalies = (values > upper) | (values < lower)
        
        return anomalies, {
            'mean': mean, 'std': std, 
            'upper_bound': upper, 'lower_bound': lower
        }
    
    @staticmethod
    def detect_iqr(values: np.ndarray, k: float = 1.5) -> Tuple[np.ndarray, Dict]:
        """IQR箱线图法"""
        if len(values) < 4:
            return np.zeros(len(values), dtype=bool), {}
        
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        
        if iqr == 0:
            median = np.median(values)
            return np.zeros(len(values), dtype=bool), {
                'q1': q1, 'q3': q3, 'iqr': 0, 'median': median, 'is_constant': True
            }
        
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        anomalies = (values < lower) | (values > upper)
        
        return anomalies, {
            'q1': q1, 'q3': q3, 'iqr': iqr, 'median': np.median(values),
            'lower_bound': lower, 'upper_bound': upper
        }
    
    @staticmethod
    def detect_mad(values: np.ndarray, threshold: float = 3.5) -> Tuple[np.ndarray, Dict]:
        """MAD中位数绝对偏差法"""
        if len(values) < 3:
            return np.zeros(len(values), dtype=bool), {}
        
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        if mad == 0:
            mad = np.mean(np.abs(values - median))
            if mad == 0:
                return np.zeros(len(values), dtype=bool), {
                    'median': median, 'mad': 0, 'is_constant': True
                }
        
        mad_scaled = 1.4826 * mad
        deviations = np.abs(values - median) / mad_scaled
        anomalies = deviations > threshold
        
        return anomalies, {
            'median': median, 'mad': mad, 'mad_scaled': mad_scaled,
            'threshold': threshold, 'std': mad_scaled
        }
    
    @staticmethod
    def detect_zscore(values: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, Dict]:
        """改进的Z-score方法（基于MAD）"""
        if len(values) < 3:
            return np.zeros(len(values), dtype=bool), {}
        
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        if mad == 0:
            return np.zeros(len(values), dtype=bool), {
                'median': median, 'mad': 0, 'is_constant': True
            }
        
        modified_z_scores = 0.6745 * (values - median) / mad
        anomalies = np.abs(modified_z_scores) > threshold
        
        return anomalies, {
            'median': median, 'mad': mad, 'threshold': threshold,
            'std': mad * 1.4826
        }
    
    @staticmethod
    def detect_percentile(values: np.ndarray, lower: float = 1, upper: float = 99) -> Tuple[np.ndarray, Dict]:
        """百分位数方法"""
        if len(values) < 10:
            return np.zeros(len(values), dtype=bool), {}
        
        lower_bound = np.percentile(values, lower)
        upper_bound = np.percentile(values, upper)
        anomalies = (values < lower_bound) | (values > upper_bound)
        
        return anomalies, {
            'lower_percentile': lower, 'upper_percentile': upper,
            'lower_bound': lower_bound, 'upper_bound': upper_bound,
            'median': np.median(values), 'std': np.std(values)
        }
    
    @staticmethod
    def detect_sudden_change(values: np.ndarray, max_change: float) -> np.ndarray:
        """检测突变（相邻值变化过大）"""
        if len(values) < 2:
            return np.zeros(len(values), dtype=bool)
        
        diffs = np.abs(np.diff(values))
        anomalies = np.zeros(len(values), dtype=bool)
        anomalies[1:] = diffs > max_change
        return anomalies


class TimeSeriesDetector:
    """时间序列异常检测器 - 考虑时序依赖性"""
    
    @staticmethod
    def detect_arima_residuals(values: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, Dict]:
        """
        ARIMA残差分析法
        原理: 用ARIMA预测，预测误差过大的为异常
        优势: 考虑时序自相关性
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            return np.zeros(len(values), dtype=bool), {'error': 'statsmodels not installed'}
        
        if len(values) < 20:
            return np.zeros(len(values), dtype=bool), {'error': 'insufficient data'}
        
        try:
            model = ARIMA(values, order=(1, 0, 1))
            fitted = model.fit()
            residuals = fitted.resid
            
            std_resid = np.std(residuals)
            if std_resid == 0:
                return np.zeros(len(values), dtype=bool), {}
            
            anomalies = np.abs(residuals) > threshold * std_resid
            
            return anomalies, {
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(std_resid),
                'aic': float(fitted.aic)
            }
        except Exception as e:
            return np.zeros(len(values), dtype=bool), {'error': str(e)}
    
    @staticmethod
    def detect_stl_residuals(values: np.ndarray, period: int = 6, 
                            threshold: float = 3.0) -> Tuple[np.ndarray, Dict]:
        """
        STL季节-趋势分解
        原理: 分解为趋势+季节+残差，残差异常为真异常
        优势: 自动处理周期性和趋势
        """
        try:
            from statsmodels.tsa.seasonal import STL
        except ImportError:
            return np.zeros(len(values), dtype=bool), {'error': 'statsmodels not installed'}
        
        if len(values) < 2 * period:
            return np.zeros(len(values), dtype=bool), {'error': f'need at least {2*period} points'}
        
        try:
            stl = STL(values, period=period, robust=True)
            result = stl.fit()
            residuals = result.resid
            
            median_resid = np.median(residuals)
            mad_resid = np.median(np.abs(residuals - median_resid))
            
            if mad_resid == 0:
                return np.zeros(len(values), dtype=bool), {}
            
            mad_scaled = 1.4826 * mad_resid
            anomalies = np.abs(residuals - median_resid) > threshold * mad_scaled
            
            return anomalies, {
                'median_residual': float(median_resid),
                'mad_residual': float(mad_resid)
            }
        except Exception as e:
            return np.zeros(len(values), dtype=bool), {'error': str(e)}


class MLDetector:
    """机器学习异常检测器"""
    
    @staticmethod
    def detect_isolation_forest(values: np.ndarray, contamination: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """
        孤立森林
        原理: 异常点更容易被孤立（划分）
        优势: 无需假设数据分布，适合高维数据
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            return np.zeros(len(values), dtype=bool), {'error': 'sklearn not installed'}
        
        if len(values) < 10:
            return np.zeros(len(values), dtype=bool), {}
        
        X = values.reshape(-1, 1)
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(X)
        anomalies = predictions == -1
        
        return anomalies, {
            'contamination': contamination,
            'n_anomalies': int(np.sum(anomalies))
        }
    
    @staticmethod
    def detect_lof(values: np.ndarray, contamination: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """
        局部离群因子 (LOF)
        原理: 基于局部密度，密度明显低于邻居的为异常
        优势: 适合密度不均匀的数据
        """
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            return np.zeros(len(values), dtype=bool), {'error': 'sklearn not installed'}
        
        if len(values) < 10:
            return np.zeros(len(values), dtype=bool), {}
        
        X = values.reshape(-1, 1)
        clf = LocalOutlierFactor(contamination=contamination)
        predictions = clf.fit_predict(X)
        anomalies = predictions == -1
        
        return anomalies, {
            'contamination': contamination,
            'n_anomalies': int(np.sum(anomalies))
        }
    
    @staticmethod
    def detect_one_class_svm(values: np.ndarray, contamination: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """
        One-Class SVM
        原理: 学习"正常数据"的边界，超出边界为异常
        优势: 适合单类别问题
        """
        try:
            from sklearn.svm import OneClassSVM
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return np.zeros(len(values), dtype=bool), {'error': 'sklearn not installed'}
        
        if len(values) < 10:
            return np.zeros(len(values), dtype=bool), {}
        
        X = values.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
        predictions = clf.fit_predict(X_scaled)
        anomalies = predictions == -1
        
        return anomalies, {
            'contamination': contamination,
            'n_anomalies': int(np.sum(anomalies))
        }


class SpatialDetector:
    """空间异常检测器 - 考虑地理位置相关性"""
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        计算两点间的地理距离（公里）
        """
        R = 6371  # 地球半径（公里）
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    
    @staticmethod
    def find_neighbors(station_idx: int, locations: np.ndarray, 
                      max_distance: float = 100,
                      max_elev_diff: float = 500) -> List[int]:
        """
        找出某站点的邻近站点（同时考虑水平距离和海拔差异）
        参数:
            station_idx: 站点索引
            locations: [[lat1, lon1, elev1], [lat2, lon2, elev2], ...]
            max_distance: 最大水平距离阈值（公里）
            max_elev_diff: 最大海拔差异阈值（米）
        """
        neighbors = []
        target_lat, target_lon, target_elev = locations[station_idx]
        for i, loc in enumerate(locations):
            if i == station_idx:
                continue
            # 1. 检查海拔差异
            elev_diff = abs(target_elev - loc[2])
            if elev_diff > max_elev_diff:
                continue
            # 2. 检查水平距离
            dist = SpatialDetector.haversine_distance(target_lat, target_lon, loc[0], loc[1])
            if dist <= max_distance:
                neighbors.append(i)
        return neighbors
    
    
    @staticmethod
    def elevation_adjusted_value(value: float, elev_diff: float, 
                                 var_type: str = 'temp') -> float:
        """
        根据海拔差异调整变量值
        气象学经验:
        - 温度: 每升高100m降低0.65°C (干绝热递减率)
        - 气压: 每升高10m降低约1.2hPa
        - 湿度: 海拔影响较小，不调整
        """
        if var_type == 'temp':
            # 温度随海拔降低: -0.65°C/100m
            return value + (elev_diff / 100) * 0.65
        elif var_type == 'bar':
            # 气压随海拔降低: -1.2hPa/10m
            return value + (elev_diff / 10) * 1.2
        else:
            # 其他变量不调整
            return value
    
    
    @staticmethod
    def detect_spatial_anomalies(
        station_data: Dict[str, Dict],  # {station_id: {var: value, lat, lon, elev}}
        variable: str,
        threshold: float = 3.0,
        max_distance: float = 100,
        min_neighbors: int = 2,
        max_elev_diff: float = 500
    ) -> Tuple[List[str], Dict]:
        """
        检测空间异常，核心循环，逐个站点对比，算出偏离度
        """
        station_ids = list(station_data.keys())
        n_stations = len(station_ids)
        if n_stations < min_neighbors + 1:
            return [], {'error': 'insufficient stations'}
        # 提取位置和值
        locations = np.array([
            [station_data[sid]['latitude'], 
             station_data[sid]['longitude'],
             station_data[sid]['elevation']]
            for sid in station_ids
        ])
        values = np.array([station_data[sid].get(variable, np.nan) for sid in station_ids])
        # 检测每个站点
        anomalous_stations = []
        details = {}
        for i, station_id in enumerate(station_ids):
            if np.isnan(values[i]):
                continue
            # 找邻近站点
            neighbor_indices = SpatialDetector.find_neighbors(i, locations, max_distance, max_elev_diff)
            if len(neighbor_indices) < min_neighbors:
                continue  # 邻居太少，无法判断
            
            # 获取邻近站点的值（考虑海拔修正）
            target_elev = locations[i, 2]
            neighbor_values_adjusted = []
            neighbor_raw_values = []
            
            for j in neighbor_indices:
                if np.isnan(values[j]):
                    continue
                
                elev_diff = locations[j, 2] - target_elev
                adjusted_val = SpatialDetector.elevation_adjusted_value(
                    values[j], elev_diff, var_type=variable
                )
                neighbor_values_adjusted.append(adjusted_val)
                neighbor_raw_values.append(values[j])
            
            if len(neighbor_values_adjusted) < min_neighbors:
                continue
            
            # 计算邻近站点的统计量
            neighbor_median = np.median(neighbor_values_adjusted)
            neighbor_mad = np.median(np.abs(np.array(neighbor_values_adjusted) - neighbor_median))
            
            if neighbor_mad == 0:
                neighbor_mad = np.std(neighbor_values_adjusted) or 1e-6
            
            # 计算偏离程度
            deviation = abs(values[i] - neighbor_median) / (1.4826 * neighbor_mad)
            
            if deviation > threshold:
                anomalous_stations.append(station_id)
                details[station_id] = {
                    'value': float(values[i]),
                    'neighbor_median': float(neighbor_median),
                    'neighbor_mad': float(neighbor_mad),
                    'deviation': float(deviation),
                    'n_neighbors': len(neighbor_values_adjusted),
                    'neighbor_ids': [station_ids[j] for j in neighbor_indices],
                    'neighbor_raw_values': [float(x) for x in neighbor_raw_values],
                    'neighbor_adj_values': [float(x) for x in neighbor_values_adjusted]
                }
                
        return anomalous_stations, details





class AnomalyDetector:
    """异常检测主控制器"""
    
    # 支持的检测方法
    AVAILABLE_METHODS = {
        # 统计方法（基于分布）
        '3sigma': '3σ规则 - 假设正态分布',
        'iqr': 'IQR箱线图法 - 鲁棒，适合偏态数据', # 不可信，弃用
        'mad': 'MAD中位数绝对偏差 - 抗噪声',  # 不可信，弃用
        'zscore': '改进Z-score - 基于MAD',
        'percentile': '百分位数法 - 定义稀有度',
        
        # 时序方法（考虑时间依赖）
        'arima': 'ARIMA残差法 - 考虑自相关性',
        'stl': 'STL分解法 - 处理趋势和周期',
        
        # 机器学习方法
        'isolation_forest': '孤立森林 - 无分布假设',
        'lof': '局部离群因子 - 基于密度',
        'ocsvm': 'One-Class SVM - 学习正常边界',
        
        # 空间方法（考虑地理位置）
        'spatial': '空间异常检测 - 基于邻近站点相关性'
    }
    
    # 检测变量配置
    DETECTION_VARS = {
        'temp_out': {'name': '温度', 'unit': '°C', 'threshold': 3, 'sudden_change': 5.0},
        'out_hum': {'name': '湿度', 'unit': '%', 'threshold': 3},
        'wind_speed': {'name': '风速', 'unit': 'km/h', 'threshold': 3},
        'bar': {'name': '气压', 'unit': 'hPa', 'threshold': 3, 'sudden_change': 10.0}
    }
    
    def __init__(self, db_path: str = 'weather_stream.db', 
                 start_time: str = None, end_time: str = None, window_hours: int = None,
                 temporal_method: str = '3sigma', spatial_method: str = 'mad', spatial_verify: bool = False):
        """
        Initialize Anomaly Detector.
        
        Args:
            db_path: Path to SQLite database
            start_time: Window start time (YYYY-MM-DD HH:MM:SS)
            end_time: Window end time (YYYY-MM-DD HH:MM:SS)
            window_hours: Window duration in hours
            temporal_method: Method for temporal detection (e.g., 'arima', '3sigma')
            spatial_method: Method for spatial fallback (e.g., 'mad')
            spatial_verify: Enable spatial cross-verification (Trend Analysis)
        """
        self.start_time = start_time
        self.end_time = end_time
        self.window_hours = window_hours
        self.temporal_method = temporal_method
        self.spatial_method = spatial_method
        self.spatial_verify = spatial_verify
        
        # Validate parameters
        if not ((start_time and end_time) or (end_time and window_hours)):
            raise ValueError("Must specify: (start_time + end_time) OR (end_time + window_hours)")
        
        # Validate temporal method
        if temporal_method not in self.AVAILABLE_METHODS:
            raise ValueError(f"Unsupported method: {temporal_method}. Available: {list(self.AVAILABLE_METHODS.keys())}")
        
        self.loader = WindowDataLoader(db_path)
        self.stat_detector = StatisticalDetector()
        self.ts_detector = TimeSeriesDetector()
        self.ml_detector = MLDetector()
    
    
    
    def verify_spatial_trend(self, station_id: str, timestamp: str, 
                           variable: str, window_minutes: int = 30) -> Dict:
        """
        高级空间验证：基于趋势相关性 (Pearson Correlation)
        逻辑:
            1. 取目标站点和邻居在 [T-window, T+window] 的数据
            2. 对数据进行线性插值填补空洞
            3. 计算目标站点与邻居的皮尔逊相关系数
            4. 如果强相关(>0.7) -> 环境变化; 弱相关(<0.3) -> 设备故障
        """
        # 1. 确定时间范围
        dt = pd.to_datetime(timestamp)
        start_dt = dt - timedelta(minutes=window_minutes)
        end_dt = dt + timedelta(minutes=window_minutes)
        
        # 2. 获取所有站点位置信息（为了找邻居）
        stations_df = self.loader.get_all_stations()
        locations = np.array([
            [row['latitude'], row['longitude'], row['elevation']]
            for _, row in stations_df.iterrows()
        ])
        station_ids = stations_df['station_id'].tolist()
        
        try:
            target_idx = station_ids.index(station_id)
        except ValueError:
            return {'error': 'station not found'}
            
        # 3. 找出邻居 (使用 SpatialDetector 的静态方法)
        neighbor_indices = SpatialDetector.find_neighbors(
            target_idx, locations, max_distance=100, max_elev_diff=500
        )
        
        if not neighbor_indices:
            return {'status': 'no_neighbors', 'correlation': 0, 'msg': '无邻居'}
            
        neighbor_ids = [station_ids[i] for i in neighbor_indices]
        
        # 4. 查询数据 (目标站点 + 邻居)
        # 构造查询所有涉及站点的数据
        all_ids = [station_id] + neighbor_ids
        placeholders = ','.join(['?'] * len(all_ids))
        
        query = f"""
            SELECT time, station_id, {variable}
            FROM observations
            WHERE station_id IN ({placeholders})
            AND time BETWEEN ? AND ?
            ORDER BY time
        """
        
        params = all_ids + [start_dt.strftime('%Y-%m-%d %H:%M:%S'), 
                          end_dt.strftime('%Y-%m-%d %H:%M:%S')]
        
        df = pd.read_sql_query(query, self.loader.conn, params=params)
        
        if df.empty:
            return {'status': 'no_data', 'correlation': 0}
            
        # 5. 数据透视与清洗
        df['time'] = pd.to_datetime(df['time'])
        pivot_df = df.pivot(index='time', columns='station_id', values=variable)
        
        # 确保目标站点在列中
        if station_id not in pivot_df.columns:
            return {'status': 'no_data', 'correlation': 0}
            
        # 6. 插值填补空洞 (关键步骤)
        # 使用时间索引进行线性插值，限制插值方向
        pivot_df = pivot_df.interpolate(method='time', limit_direction='both', limit=2)
        
        # 再次清理仍为NaN的行（插值失败的）
        pivot_df.dropna(inplace=True)
        
        if len(pivot_df) < 5:  # 数据点太少无法计算相关性
            return {'status': 'insufficient_points', 'correlation': 0}
            
        # 7. 计算相关系数
        target_series = pivot_df[station_id]
        correlations = []
        valid_neighbors = []
        
        for nid in neighbor_ids:
            if nid in pivot_df.columns:
                # 计算皮尔逊系数
                corr = target_series.corr(pivot_df[nid])
                if not np.isnan(corr):
                    correlations.append(corr)
                    valid_neighbors.append(nid)
        
        if not correlations:
            return {'status': 'no_valid_correlations', 'correlation': 0}
            
        # 取相关系数的中位数或最大值作为最终判定依据
        # 这里取中位数比较稳健
        median_corr = np.median(correlations)
        max_corr = np.max(correlations)
        
        return {
            'status': 'success',
            'median_corr': median_corr,
            'max_corr': max_corr,
            'n_neighbors': len(correlations),
            'valid_neighbors': valid_neighbors,
            'is_trend_consistent': median_corr > 0.6 or max_corr > 0.8
        }




    def detect_station(self, station_id: str) -> Dict:
        """Detect anomalies for a single station."""
        # Load data
        df = self.loader.get_window_data(station_id, 
                                         start_time=self.start_time,
                                         end_time=self.end_time, 
                                         window_hours=self.window_hours)
        
        # Validate data
        if df.empty:
            return {'station_id': station_id, 'status': 'no_data', 'message': 'No Data'}
        if len(df) < 3:
            return {'station_id': station_id, 'status': 'insufficient_data', 
                   'message': f'Insufficient Data ({len(df)} points)'}
        
        # Initialize result structure
        result = {
            'station_id': station_id,
            'window_start': str(df['time'].min()),
            'window_end': str(df['time'].max()),
            'data_count': len(df),
            'anomalies': {},
            'has_anomaly': False
        }
        
        # Detect each variable
        for var, config in self.DETECTION_VARS.items():
            anomaly_info = self._detect_variable(df, var, config)
            if anomaly_info:
                # Spatial Verification (if enabled)
                if self.spatial_verify:
                    for record in anomaly_info['anomaly_records']:
                        # === Advanced: Spatial Trend Verification ===
                        # Use same window length as temporal detection, look backwards only
                        trend_res = self.verify_spatial_trend(
                            station_id=station_id,
                            timestamp=record['time'],
                            variable=var,
                            window_minutes=self.window_hours * 60
                        )
                        
                        if trend_res.get('status') == 'success':
                            corr = trend_res['median_corr']
                            if trend_res['is_trend_consistent']:
                                record['type'] = 'weather_event'
                                record['label'] = '🌧️ Extreme Weather / Env Change'
                                record['desc'] = f"Trend Consistent (Corr: {corr:.2f}, {trend_res['n_neighbors']} neighbors)"
                            elif corr < 0.3:
                                record['type'] = 'critical_failure'
                                record['label'] = '🔴 Device Failure (High Confidence)'
                                record['desc'] = f"Trend Inconsistent (Corr: {corr:.2f}, erratic)"
                            else:
                                # Grey area (0.3 ~ 0.6)
                                record['type'] = 'warning'
                                record['label'] = '⚠️ Suspected Anomaly (Manual Check)'
                                record['desc'] = f"Weak Correlation (Corr: {corr:.2f})"
                        else:
                            # Fallback to static snapshot comparison if trend verification fails
                            spatial_res = self.detect_spatial_anomalies(
                                timestamp=record['time'], 
                                variable_filter=var, 
                                method=self.spatial_method,
                                verbose=False
                            )
                            
                            # Check if it's an anomaly in snapshot
                            if var in spatial_res['variables'] and \
                               station_id in spatial_res['variables'][var]['anomalous_stations']:
                                record['label'] = '🔴 Device Failure (Static Check)'
                                record['desc'] = f"High Deviation (Trend Skipped: {trend_res.get('status', 'unknown')})"
                            else:
                                record['label'] = '🌧️ Extreme Weather (Static Check)'
                                record['desc'] = f"Low Deviation (Trend Skipped: {trend_res.get('status', 'unknown')})"
                
                # --- DEBUG: Print full sequence for context ---
                print(f"\n{'='*40} DEBUG: Sequence Data {'='*40}")
                print(f"Station: {station_id}, Variable: {var}")
                print(f"Window: {df['time'].min()} ~ {df['time'].max()}")
                print("-" * 100)
                
                # Extract time series
                times = df['time'].dt.strftime('%H:%M').values
                vals = df[var].values
                
                # Print
                for t, v in zip(times, vals):
                    mark = ""
                    # Check if this point is flagged
                    for rec in anomaly_info['anomaly_records']:
                        if rec['time'].endswith(t + ":00"):
                            mark = f"<--- ⚠️  Anomaly ({rec.get('deviation', 0):.1f}σ)"
                            break
                    print(f"{t} | {v:8.2f} {config['unit']} {mark}")
                print(f"{'='*100}\n")
                # --------------------------------------------

                result['anomalies'][var] = anomaly_info
                result['has_anomaly'] = True
        
        return result
    
    def _detect_variable(self, df: pd.DataFrame, var: str, config: Dict) -> Optional[Dict]:
        """检测单个变量的异常"""
        if var not in df.columns:
            return None
        
        values = df[var].values
        if np.all(np.isnan(values)):
            return None
        
        # 根据方法选择检测器
        if self.temporal_method == '3sigma':
            anomaly_mask, stats = self.stat_detector.detect_3sigma(values, config['threshold'])
        elif self.temporal_method == 'iqr':
            anomaly_mask, stats = self.stat_detector.detect_iqr(values, k=1.5)
        elif self.temporal_method == 'mad':
            anomaly_mask, stats = self.stat_detector.detect_mad(values, threshold=3.5)
        elif self.temporal_method == 'zscore':
            anomaly_mask, stats = self.stat_detector.detect_zscore(values, threshold=3.0)
        elif self.temporal_method == 'percentile':
            anomaly_mask, stats = self.stat_detector.detect_percentile(values, lower=1, upper=99)
        elif self.temporal_method == 'arima':
            anomaly_mask, stats = self.ts_detector.detect_arima_residuals(values, threshold=3.0)
        elif self.temporal_method == 'stl':
            anomaly_mask, stats = self.ts_detector.detect_stl_residuals(values, period=6, threshold=3.0)
        elif self.temporal_method == 'isolation_forest':
            anomaly_mask, stats = self.ml_detector.detect_isolation_forest(values, contamination=0.1)
        elif self.temporal_method == 'lof':
            anomaly_mask, stats = self.ml_detector.detect_lof(values, contamination=0.1)
        elif self.temporal_method == 'ocsvm':
            anomaly_mask, stats = self.ml_detector.detect_one_class_svm(values, contamination=0.1)
        else:
            # 默认3sigma
            anomaly_mask, stats = self.stat_detector.detect_3sigma(values, config['threshold'])
        
        # 检查是否有错误
        if 'error' in stats:
            return None
        
        # 突变检测（作为额外检测，如果配置了）
        if 'sudden_change' in config:
            sudden_mask = self.stat_detector.detect_sudden_change(values, config['sudden_change'])
            anomaly_mask = anomaly_mask | sudden_mask
        
        # 如果没有异常，返回None
        if not np.any(anomaly_mask):
            return None
        
        # 构建异常记录
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_records = []
        
        for idx in anomaly_indices:
            record = {
                'time': str(df.iloc[idx]['time']),
                'value': float(values[idx])
            }
            
            # 计算偏离度（根据方法）
            if self.temporal_method in ['3sigma', 'zscore', 'arima', 'stl'] and stats.get('std', 0) > 0:
                mean_val = stats.get('mean', stats.get('median', 0))
                record['deviation'] = float(abs(values[idx] - mean_val) / stats['std'])
            elif self.temporal_method == 'iqr' and stats.get('iqr', 0) > 0:
                record['deviation'] = float(abs(values[idx] - stats['median']) / stats['iqr'])
            elif self.temporal_method == 'mad' and stats.get('mad_scaled', 0) > 0:
                record['deviation'] = float(abs(values[idx] - stats['median']) / stats['mad_scaled'])
            else:
                record['deviation'] = 0.0
            
            anomaly_records.append(record)
        
        return {
            'name': config['name'],
            'unit': config['unit'],
            'count': int(np.sum(anomaly_mask)),
            'method': self.temporal_method,
            'statistics': {k: float(v) for k, v in stats.items() if k not in ['is_constant', 'error']},
            'anomaly_records': anomaly_records
        }
    
    def detect_all_stations(self) -> List[Dict]:
        """检测所有站点"""
        stations_df = self.loader.get_all_stations()
        results = []
        
        for _, row in stations_df.iterrows():
            result = self.detect_station(row['station_id'])
            result['station_name'] = row['station_name_en']
            results.append(result)
        
        return results
    
    def detect_spatial_anomalies(self, timestamp: str = None, 
                                max_distance: float = 100,
                                threshold: float = 3.0,
                                max_elev_diff: float = 500,
                                variable_filter: str = None,
                                method: str = 'mad',
                                verbose: bool = True) -> Dict:
        """
        空间异常检测
        参数:
            method: 空间检测方法 (支持 'mad', 'zscore', '3sigma')
        """
        # 获取所有站点信息
        stations_df = self.loader.get_all_stations()
        # 获取指定时刻的数据
        if timestamp is None:
            # 使用窗口结束时刻
            timestamp = self.end_time
        # 查询所有站点在该时刻的数据
        query = """
            SELECT o.station_id, o.temp_out, o.out_hum, o.wind_speed, o.bar,
                   s.latitude, s.longitude, s.elevation, s.station_name_en
            FROM observations o
            JOIN stations s ON o.station_id = s.station_id
            WHERE o.time = ?
        """
        df = pd.read_sql_query(query, self.loader.conn, params=(timestamp,))
        if df.empty:
            return {'error': 'no data at specified time', 'timestamp': timestamp}
        
        # 准备站点数据字典
        station_data = {}
        for _, row in df.iterrows():
            station_data[row['station_id']] = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'elevation': row['elevation'],
                'temp_out': row['temp_out'],
                'out_hum': row['out_hum'],
                'wind_speed': row['wind_speed'],
                'bar': row['bar'],
                'station_name': row['station_name_en']
            }
        
        # 对每个变量进行空间检测
        results = {
            'timestamp': timestamp,
            'n_stations': len(station_data),
            'max_distance': max_distance,
            'max_elev_diff': max_elev_diff,
            'threshold': threshold,
            'variables': {}
        }
        
        spatial_detector = SpatialDetector()
        
        # --- 打印一次邻居关系 (仅基于地理位置和海拔，不依赖变量) ---
        station_ids = list(station_data.keys())
        n_stations = len(station_ids)
        
        if n_stations > 1 and verbose:
            locations = np.array([
                [station_data[sid]['latitude'], 
                 station_data[sid]['longitude'],
                 station_data[sid]['elevation']]
                for sid in station_ids
            ])
            
            print(f"\n{'='*80}")
            print(f"Spatial Neighbor Analysis (Max Dist: {max_distance}km, Max Elev Diff: {max_elev_diff}m)")
            print(f"{'='*80}")
            
            for i, station_id in enumerate(station_ids):
                neighbor_indices = spatial_detector.find_neighbors(i, locations, max_distance, max_elev_diff)
                
                if not neighbor_indices:
                    print(f"Station {station_id}: No neighbors found")
                    continue
                
                print(f"Station {station_id} ({station_data[station_id]['station_name']}) - {len(neighbor_indices)} neighbors:")
                target_elev = locations[i, 2]
                
                for idx in neighbor_indices:
                    nid = station_ids[idx]
                    n_name = station_data[nid]['station_name']
                    n_elev = locations[idx, 2]
                    dist = spatial_detector.haversine_distance(
                        locations[i, 0], locations[i, 1], 
                        locations[idx, 0], locations[idx, 1]
                    )
                    elev_diff = n_elev - target_elev
                    print(f"  -> {nid} ({n_name}): Dist {dist:.1f}km, Elev {n_elev:.0f}m (Diff: {elev_diff:+.0f}m)")
            print(f"{'='*80}\n")

        # --- 执行变量检测 ---
        for var, config in self.DETECTION_VARS.items():
            # 如果指定了变量过滤器，跳过不相关的变量
            if variable_filter and var != variable_filter:
                continue
                
            anomalous_stations, details = spatial_detector.detect_spatial_anomalies(
                station_data, var, threshold, max_distance, min_neighbors=2,
                max_elev_diff=max_elev_diff
            )
            
            if anomalous_stations:
                results['variables'][var] = {
                    'name': config['name'],
                    'unit': config['unit'],
                    'anomalous_stations': anomalous_stations,
                    'details': details
                }
        
        return results
    
    def close(self):
        self.loader.close()





class ReportGenerator:
    """Generates analysis reports."""
    @staticmethod
    def generate_text_report(results: List[Dict], window_info: str = None, method: str = '3sigma') -> str:
        """Generate text report."""
        method_names = {
            '3sigma': '3-Sigma Rule', 'iqr': 'IQR Method', 'mad': 'MAD (Median Absolute Deviation)',
            'zscore': 'Modified Z-Score', 'percentile': 'Percentile',
            'arima': 'ARIMA Residuals', 'stl': 'STL Decomposition',
            'isolation_forest': 'Isolation Forest', 'lof': 'Local Outlier Factor', 'ocsvm': 'One-Class SVM'
        }
        
        lines = [
            "ANOMALY DETECTION REPORT",
            f"Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Window Info: {window_info}" if window_info else "",
            f"Method: {method_names.get(method, method)}",
            ""
        ]
        
        # Statistics
        total = len(results)
        stations_with_anomaly = [r for r in results if r.get('has_anomaly', False)]
        anomaly_count = len(stations_with_anomaly)
        
        # Breakdown by anomaly type
        type_counts = {'critical_failure': 0, 'weather_event': 0, 'warning': 0}
        
        for r in stations_with_anomaly:
            for var_info in r['anomalies'].values():
                for rec in var_info['anomaly_records']:
                    atype = rec.get('type', 'unknown')
                    if atype in type_counts:
                        type_counts[atype] += 1
        
        lines.extend([
            f"Total Stations: {total}",
            f"Anomalous Stations: {anomaly_count}",
            f"Normal Stations: {total - anomaly_count}",
            "",
            "Anomaly Breakdown:",
            f"  🔴 Device Failures: {type_counts['critical_failure']}",
            f"  🌧️ Weather Events: {type_counts['weather_event']} (Ignorable)",
            f"  ⚠️ Suspected:      {type_counts['warning']}",
            "",
            "=" * 100,
            ""
        ])
        
        # Detailed Info
        for result in results:
            if result.get('has_anomaly'):
                lines.extend(ReportGenerator._format_station_anomalies(result))
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_station_anomalies(result: Dict) -> List[str]:
        """Format anomalies for a single station."""
        lines = [
            f"[Station: {result['station_id']} - {result.get('station_name', 'Unknown')}]",
            f"  Window: {result['window_start']} ~ {result['window_end']}",
            f"  Data Points: {result['data_count']}",
            ""
        ]
        
        for var, info in result['anomalies'].items():
            stats = info['statistics']
            lines.append(f"  ⚠️  {info['name']} Anomaly:")
            lines.append(f"      Count: {info['count']}")
            lines.append(f"      Method: {info.get('method', 'unknown')}")
            
            # Stats
            if 'mean' in stats and 'std' in stats:
                lines.append(f"      Stats: Mean={stats['mean']:.2f}{info['unit']}, Std={stats['std']:.2f}{info['unit']}")
            elif 'median' in stats and 'std' in stats:
                lines.append(f"      Stats: Median={stats['median']:.2f}{info['unit']}, Std={stats['std']:.2f}{info['unit']}")
            elif 'median' in stats and 'iqr' in stats:
                lines.append(f"      Stats: Median={stats['median']:.2f}{info['unit']}, IQR={stats['iqr']:.2f}{info['unit']}")
            
            # Normal Range
            if 'lower_bound' in stats and 'upper_bound' in stats:
                lines.append(f"      Normal Range: [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}] {info['unit']}")
            
            # Records
            for record in info['anomaly_records']:
                line = f"      • {record['time']}: {record['value']:.2f}{info['unit']} (Dev: {record['deviation']:.1f}σ)"
                if 'label' in record:
                    line += f" -> {record['label']}"
                lines.append(line)
                if 'desc' in record:
                     lines.append(f"        └─ Diag: {record['desc']}")
            lines.append("")
        
        return lines
    
    @staticmethod
    def save_json_report(results: List[Dict], window_info: dict = None, filename: str = None) -> str:
        """Save JSON report."""
        if filename is None:
            filename = f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            'detection_time': datetime.now().isoformat(),
            'results': results
        }
        
        if window_info:
            report_data['window_info'] = window_info
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return filename





def main():
    parser = argparse.ArgumentParser(
        description='Weather Data Anomaly Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Statistical Method
  python anomaly_detector.py --end "2025-11-20 16:00:00" --window 6 --temporal-method 3sigma
  
  # Time-Series Method (Recommended)
  python anomaly_detector.py --end "2025-11-20 16:00:00" --window 6 --temporal-method arima --spatial-verify
  
  # Machine Learning Method
  python anomaly_detector.py --end "2025-11-20 16:00:00" --window 6 --temporal-method lof
  
  # List all methods
  python anomaly_detector.py --list-methods
        """
    )
    
    parser.add_argument('--db', default='weather_stream.db', help='Path to SQLite DB')
    parser.add_argument('--start', help='Start time (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end', help='End time (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--window', type=int, help='Window size (hours)')
    parser.add_argument('--temporal-method', default='3sigma',
                       choices=list(AnomalyDetector.AVAILABLE_METHODS.keys()),
                       help='Temporal method (default: 3sigma)')
    parser.add_argument('--spatial-method', default='mad',
                       choices=['mad', 'zscore', '3sigma'],
                       help='Spatial fallback method (default: mad)')
    parser.add_argument('--station', help='Target station ID')
    parser.add_argument('--save', action='store_true', help='Save result to JSON')
    parser.add_argument('--spatial-verify', action='store_true', 
                       help='Enable spatial trend verification')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    parser.add_argument('--list-methods', action='store_true', help='List all methods')
    
    args = parser.parse_args()
    
    # 列出检测方法
    if args.list_methods:
        print("\n" + "="*80)
        print("Available Detection Methods:")
        print("="*80)
        for method, desc in AnomalyDetector.AVAILABLE_METHODS.items():
            print(f"  {method:20s} - {desc}")
        print("="*80 + "\n")
        return
    
    # Validate args
    if not ((args.start and args.end) or (args.end and args.window)):
        parser.error("Must specify: (--start AND --end) OR (--end AND --window)")
    
    # Prepare window info
    if args.start and args.end:
        window_info_str = f"{args.start} ~ {args.end}"
        window_info_dict = {'start_time': args.start, 'end_time': args.end}
    else:
        window_info_str = f"End at {args.end} (Lookback {args.window} hours)"
        window_info_dict = {'end_time': args.end, 'window_hours': args.window}
    
    window_info_dict['temporal_method'] = args.temporal_method
    
    # 执行检测
    detector = AnomalyDetector(
        db_path=args.db, 
        start_time=args.start,
        end_time=args.end,
        window_hours=args.window,
        temporal_method=args.temporal_method,
        spatial_method=args.spatial_method,
        spatial_verify=args.spatial_verify
    )
    
    try:
        # 检测
        if args.station:
            results = [detector.detect_station(args.station)]
        else:
            results = detector.detect_all_stations()
        
        # 生成报告
        report = ReportGenerator.generate_text_report(results, window_info_str, method=args.temporal_method)
        
        if not args.quiet:
            print(report)
        
        # 保存JSON
        if args.save:
            filename = ReportGenerator.save_json_report(results, window_info_dict)
            print(f"\n 检测结果已保存到: {filename}")
    
    finally:
        detector.close()


if __name__ == '__main__':
    main()


## 用统计方法 - arima，isolation_forest，3sigma
# python anomaly_detector.py --end "2025-11-22 17:00:00" --window 6 --temporal-method arima --spatial-verify
# python anomaly_detector.py --end "2025-11-22 17:00:00" --window 6 --temporal-method isolation_forest --spatial-verify