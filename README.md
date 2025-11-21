# 🌦️ Real-Time Weather Anomaly Detection System

Real-time monitoring and anomaly detection for 14 meteorological stations from the National Observatory of Athens (NOA).

---

## 🚀 Quick Start

### 1. Start Data Collection

```bash
cd /data/qwang/q/datagem/stream_detection
source ~/software/miniconda3/bin/activate datagem

# Start background collector (every 10 minutes)
bash manage_collector.sh start
```

### 2. View Data

```bash
# View latest data
python view_data.py --realtime

# View specific station
python view_data.py --station heraclion --latest 20
```

### 3. Anomaly Detection

```bash
# Temporal anomaly detection (single station, time window)
python anomaly_detector.py --end "2025-11-21 02:00:00" --window 6 --method mad

# Spatial anomaly detection (all stations, single timestamp)
python spatial_anomaly_detector.py --time "2025-11-21 02:00:00"
```

---

## 🔧 System Components

### Data Collection
- **`streaming_collector_sqlite.py`** - Fetches data every 10 minutes from GeoJSON source
- **`weather_stream.db`** - SQLite database storing all historical data
- **`manage_collector.sh`** - Process management (start/stop/status)

### Anomaly Detection
- **`anomaly_detector.py`** - Temporal anomaly detection (11 methods)
- **`spatial_anomaly_detector.py`** - Spatial anomaly detection
- **`view_data.py`** - Data query and export tool

---

## 📊 Detection Methods

### 🕐 Temporal Methods
Detect anomalies for **a single station across time window**

| Method | Description | Use Case |
|--------|-------------|----------|
| **3sigma** | 3σ rule, assumes normal distribution | Extreme outliers only |
| **mad** ⭐ | Median Absolute Deviation, most robust | **Recommended for weather data** |
| **iqr** | Interquartile Range (boxplot) | Exploratory analysis |
| **zscore** | Modified Z-score (MAD-based) | Similar to MAD |
| **arima** | ARIMA residual analysis | Time series autocorrelation |
| **stl** | Seasonal-Trend decomposition | Periodic data |
| **isolation_forest** | Machine learning method | Complex patterns |
| **lof** | Local Outlier Factor | Non-uniform density |

**Recommended**:
- Daily monitoring: `--method mad` (balanced sensitivity & robustness)
- Critical alerts: `--method 3sigma` (extreme events only)

### 🌍 Spatial Methods
Detect anomalies **across stations at the same timestamp**

**Principle**:
1. Calculate geographic distance between stations (Haversine formula)
2. Find neighboring stations (default: within 100km)
3. Adjust for elevation differences (temp: -0.65°C/100m, pressure: -1.2hPa/10m)
4. Flag if station value deviates significantly from neighbors' median

**Advantages**:
- Distinguish sensor faults from real extreme weather
- Sensor fault: only this station anomalous, neighbors normal
- Extreme weather: both station and neighbors anomalous

---

## 💡 Usage Examples

### Scenario 1: Daily Monitoring

```bash
python anomaly_detector.py \
  --end "2025-11-21 02:00:00" \
  --window 6 \
  --method mad
```

### Scenario 2: Compare Methods

```bash
# Conservative (extreme only)
python anomaly_detector.py --end "2025-11-21 02:00:00" --window 6 --method 3sigma
# Result: 1 anomalous station

# Sensitive (more detections)
python anomaly_detector.py --end "2025-11-21 02:00:00" --window 6 --method iqr
# Result: 9 anomalous stations

# Balanced (recommended)
python anomaly_detector.py --end "2025-11-21 02:00:00" --window 6 --method mad
# Result: 5 anomalous stations
```

### Scenario 3: Spatial Validation

```bash
# Step 1: Temporal detection finds anomaly
python anomaly_detector.py --end "2025-11-21 02:00:00" --window 6 --method mad
# Output: heraclion station wind speed 24.10km/h anomalous

# Step 2: Spatial validation (sensor fault or real weather?)
python spatial_anomaly_detector.py --time "2025-11-21 02:00:00"
# If only heraclion anomalous → sensor fault
# If neighbors also anomalous → extreme weather
```

### Scenario 4: Save Results

```bash
python anomaly_detector.py \
  --end "2025-11-21 02:00:00" \
  --window 6 \
  --method mad \
  --save
# Generates: anomaly_report_20251121_023456.json
```

### Scenario 5: Batch Detection

```bash
#!/bin/bash
for hour in 00 06 12 18; do
  python anomaly_detector.py \
    --end "2025-11-21 ${hour}:00:00" \
    --window 6 \
    --method mad \
    --save \
    --quiet
  echo "✓ Done: ${hour}:00"
done
```

---

## 📁 Files

### Core Files
```
streaming_collector_sqlite.py  - Data collector
weather_stream.db              - SQLite database
manage_collector.sh            - Management script
anomaly_detector.py            - Temporal detection (11 methods)
spatial_anomaly_detector.py    - Spatial detection
view_data.py                   - Data query tool
timeseries_anomaly_detector.py - Algorithm library
README.md                      - This document
.gitignore                     - Git ignore config
```

### Archived (ignore/)
```
ignore/
├── streaming_collector_timescale.py    - TimescaleDB version (deprecated)
├── streaming_anomaly_detector_timescale.py
├── TIMESCALEDB_SETUP_GUIDE.md
└── ...
```

---

## 🔍 FAQ

**Q1: Where is data stored?**  
A: `weather_stream.db` SQLite database, updated in real-time

**Q2: View latest data?**  
```bash
python view_data.py --realtime
```

**Q3: Stop data collection?**  
```bash
bash manage_collector.sh stop
```

**Q4: Which detection method?**  
- **Daily**: `mad` - balanced sensitivity & robustness
- **Conservative**: `3sigma` - extreme anomalies only
- **Exploratory**: `iqr` - high sensitivity

**Q5: Temporal vs Spatial?**  
- **Temporal**: One station, different times, detect "if anomalous"
- **Spatial**: Multiple stations, same time, detect "who is anomalous"
- **Best**: Use temporal first, then spatial validation

**Q6: Sensor fault vs Extreme weather?**  
```bash
# Step 1: Temporal detection
python anomaly_detector.py --end "TIME" --window 6 --method mad

# Step 2: Spatial validation
python spatial_anomaly_detector.py --time "TIME"
# Only station A anomalous → sensor fault
# Station A + neighbors anomalous → extreme weather
```

---

## 📊 Data Info

**Source**: https://stratus.meteo.noa.gr/data/stations/latestValues_Datagems.geojson

**Stations**: 14 DataGEMS weather stations

**Variables**:
- Temperature (temp_out, hi_temp, low_temp)
- Humidity (out_hum)
- Pressure (bar)
- Wind (wind_speed, wind_dir, hi_speed)
- Rain (rain)
- Location (latitude, longitude, elevation)

**Update Frequency**: Every 10 minutes

**Storage**: All historical data since 2024-11-20

---

## 🛠️ Tech Stack

- **Python 3.x** - Main language
- **SQLite** - Lightweight database
- **pandas** - Data processing
- **numpy** - Numerical computation
- **statsmodels** (optional) - ARIMA/STL models
- **scikit-learn** (optional) - ML methods

---

## 📞 Contact

- **Project Path**: `/data/qwang/q/datagem/stream_detection/`
- **Server**: `qwang@172.27.96.38` (internal) / `193.48.200.25` (public)
- **Data Source**: National Observatory of Athens (NOA)

---

## 📝 Changelog

**2024-11-21**
- ✅ Added spatial anomaly detection
- ✅ Implemented 11 temporal detection methods
- ✅ Simplified documentation
- ✅ Switched from TimescaleDB to SQLite

**2024-11-20**
- ✅ Completed data collection system
- ✅ Basic anomaly detection (3σ, MAD, IQR)

---

**Last Updated**: 2024-11-21  
**Version**: v2.0
