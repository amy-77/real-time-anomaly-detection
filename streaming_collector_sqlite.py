"""
SQLite-based streaming ingestion and storage system
===================================================

Key characteristics:
- Persist real-time meteorological data with built-in SQLite
- Lightweight, no extra services required
- Optimized for sliding-window queries
- Automatically creates essential indexes

Author: Weather Anomaly Detection Team
Date: 2025-11-20
"""

import sqlite3
import requests
import json
import time
import argparse
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streaming_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SQLiteDatabase:
    """
    SQLite database manager tailored for time-series weather data.
    """
    def __init__(self, db_path: str = 'weather_stream.db'):
        """
        Initialize the database connection.

        Parameters:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._init_database()
    
    def _connect(self):
        """Establish a database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # use dict-style cursor
            logger.info(f"Connected to SQLite database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise
    
    def _init_database(self):
        """Create tables and indexes if they do not exist."""
        try:
            cursor = self.conn.cursor()
            # 1. Create station metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stations (
                    station_id TEXT PRIMARY KEY,
                    station_name_en TEXT,
                    station_name_gr TEXT,
                    latitude REAL,
                    longitude REAL,
                    elevation REAL,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("✓ Station table ready")
            
            # 2. Create observations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TIMESTAMP NOT NULL,
                    station_id TEXT NOT NULL,
                    temp_out REAL,
                    hi_temp REAL,
                    low_temp REAL,
                    out_hum REAL,
                    bar REAL,
                    rain REAL,
                    wind_speed REAL,
                    wind_dir REAL,
                    wind_dir_str TEXT,
                    hi_speed REAL,
                    hi_dir REAL,
                    hi_dir_str TEXT,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(time, station_id)
                )
            """)
            logger.info("✓ Observation table ready")
            
            # 3. Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_observations_station_time 
                ON observations (station_id, time DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_observations_time 
                ON observations (time DESC)
            """)
            logger.info("✓ Indexes ready")
            
            # 4. Create collection log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collection_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT,
                    stations_count INTEGER,
                    observations_count INTEGER,
                    message TEXT
                )
            """)
            logger.info("✓ Collection log table ready")
            
            self.conn.commit()
            logger.info("✓ Database initialized")
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def insert_station(self, station_id: str, info: Dict) -> bool:
        """
        Insert or update station metadata.
        Parameters:
            station_id: Unique station identifier.
            info: Metadata dictionary with coordinates and names.
        Returns:
            True if successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO stations (
                    station_id, station_name_en, station_name_gr,
                    latitude, longitude, elevation, first_seen, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(station_id) DO UPDATE SET
                    station_name_en = excluded.station_name_en,
                    station_name_gr = excluded.station_name_gr,
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    elevation = excluded.elevation,
                    last_seen = CURRENT_TIMESTAMP
            """, (
                station_id, info.get('station_name_en'), info.get('station_name_gr'),
                info.get('latitude'), info.get('longitude'), info.get('elevation')
            ))
            
            self.conn.commit()
            return True
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to upsert station info: {e}")
            return False
    
    def insert_observations_batch(self, observations: List[Tuple]) -> int:
        """
        Bulk insert observation rows.
        Parameters:
            observations: List of row tuples.
        Returns:
            Number of inserted rows.
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.executemany("""
                INSERT OR IGNORE INTO observations (
                    time, station_id, temp_out, hi_temp, low_temp, out_hum,
                    bar, rain, wind_speed, wind_dir, wind_dir_str,
                    hi_speed, hi_dir, hi_dir_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, observations)
            
            self.conn.commit()
            return cursor.rowcount
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to insert observation batch: {e}")
            return 0
    
    def log_collection(self, status: str, stations_count: int, obs_count: int, message: str = ""):
        """Persist collection metadata for auditing."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO collection_log (status, stations_count, observations_count, message)
                VALUES (?, ?, ?, ?)
            """, (status, stations_count, obs_count, message))
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to write collection log: {e}")
    
    def get_station_window(self, station_id: str, window_size: int = 6) -> List[Dict]:
        """
        Retrieve the latest N rows for a specific station (sliding window).

        Parameters:
            station_id: Station identifier.
            window_size: Number of rows to return.

        Returns:
            List of observation dictionaries.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT 
                    strftime('%s', time) as timestamp,
                    time,
                    station_id,
                    temp_out, hi_temp, low_temp, out_hum,
                    bar, rain, wind_speed, wind_dir, wind_dir_str,
                    hi_speed, hi_dir, hi_dir_str
                FROM observations
                WHERE station_id = ?
                ORDER BY time DESC
                LIMIT ?
            """, (station_id, window_size))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except sqlite3.Error as e:
            logger.error(f"Failed to query window data: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Return database statistics such as counts and time range."""
        try:
            cursor = self.conn.cursor()
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) as count FROM stations")
            stations_count = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM observations")
            observations_count = cursor.fetchone()['count']
            
            # Time coverage
            cursor.execute("""
                SELECT 
                    MIN(time) as earliest,
                    MAX(time) as latest
                FROM observations
            """)
            time_range = cursor.fetchone()
            
            # Last collection record
            cursor.execute("""
                SELECT * FROM collection_log 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            last_collection = cursor.fetchone()
            
            # Database size
            db_size = Path(self.db_path).stat().st_size / (1024 * 1024)  # MB
            
            return {
                'stations_count': stations_count,
                'observations_count': observations_count,
                'earliest_data': time_range['earliest'] if time_range['earliest'] else None,
                'latest_data': time_range['latest'] if time_range['latest'] else None,
                'last_collection': dict(last_collection) if last_collection else None,
                'database_size_mb': f"{db_size:.2f} MB"
            }
            
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch statistics: {e}")
            return {}
    
    def close(self):
        """Close the SQLite connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


class StreamingCollector:
    """
    Streaming collector that continuously ingests GeoJSON data into SQLite.
    """
    
    def __init__(self, 
                 geojson_url: str,
                 db: SQLiteDatabase,
                 interval_seconds: int = 600):
        """
        Parameters:
            geojson_url: Data source URL.
            db: SQLiteDatabase instance.
            interval_seconds: Collection interval in seconds.
        """
        self.geojson_url = geojson_url
        self.db = db
        self.interval_seconds = interval_seconds
        self.running = False
        self.thread = None
    
    def fetch_and_store(self) -> Tuple[bool, str]:
        """
        Fetch and store one batch of data.

        Returns:
            Tuple of (success flag, message).
        """
        try:
            logger.info(f"Fetching data from {self.geojson_url} ...")
            response = requests.get(self.geojson_url, timeout=30)
            response.raise_for_status()
            
            geojson_data = response.json()
            features = geojson_data.get('features', [])
            
            if not features:
                msg = "No features returned by GeoJSON feed"
                logger.warning(msg)
                self.db.log_collection('WARNING', 0, 0, msg)
                return False, msg
            
            # Prepare batched inserts
            station_info_list = []
            observations_list = []
            
            for feature in features:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                station_id = props.get('station_file', props.get('fid', ''))
                if not station_id:
                    continue
                
                # Station metadata
                station_info = {
                    'station_id': station_id,
                    'station_name_en': props.get('station_name_en'),
                    'station_name_gr': props.get('station_name_gr'),
                    'latitude': coords[1],
                    'longitude': coords[0],
                    'elevation': coords[2] if len(coords) > 2 else None
                }
                station_info_list.append(station_info)
                
                # Observations
                timestamp = props.get('ts', 0)
                if timestamp > 0:
                    time_dt = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    obs = (
                        time_dt, station_id,
                        props.get('temp_out'), props.get('hi_temp'), props.get('low_temp'),
                        props.get('out_hum'), props.get('bar'), props.get('rain'),
                        props.get('wind_speed'), props.get('wind_dir'), props.get('wind_dir_str'),
                        props.get('hi_speed'), props.get('hi_dir'), props.get('hi_dir_str')
                    )
                    observations_list.append(obs)
            
            # Store stations
            stations_count = 0
            for info in station_info_list:
                if self.db.insert_station(info['station_id'], info):
                    stations_count += 1
            
            # Store observations
            observations_count = self.db.insert_observations_batch(observations_list)
            
            msg = f"Stored {observations_count} observations across {stations_count} stations"
            logger.info(msg)
            self.db.log_collection('SUCCESS', stations_count, observations_count, msg)
            
            return True, msg
            
        except requests.exceptions.RequestException as e:
            msg = f"Network request failed: {e}"
            logger.error(msg)
            self.db.log_collection('ERROR', 0, 0, msg)
            return False, msg
        except Exception as e:
            msg = f"Failed to process data: {e}"
            logger.error(msg)
            self.db.log_collection('ERROR', 0, 0, msg)
            return False, msg
    
    def start(self):
        """Start the background collection thread."""
        if self.running:
            logger.warning("Collector already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.thread.start()
        logger.info("Streaming collector started")
    
    def stop(self):
        """Stop the background collection loop."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Streaming collector stopped")
    
    def _collection_loop(self):
        """Internal loop executed by the background thread."""
        logger.info(f"Collection loop started, interval: {self.interval_seconds} seconds")
        
        while self.running:
            cycle_start = time.time()
            
            # Run collection
            success, msg = self.fetch_and_store()
            
            # Determine sleep duration for next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.interval_seconds - elapsed)
            
            if self.running and sleep_time > 0:
                next_collection = datetime.now() + timedelta(seconds=sleep_time)
                logger.info(f"Next collection: {next_collection.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Sleep in 1s chunks to respond to stop signals quickly
                for _ in range(int(sleep_time)):
                    if not self.running:
                        break
                    time.sleep(1)


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='SQLite-based streaming weather data collector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start continuous collection (10 minutes interval)
  python streaming_collector_sqlite.py --continuous

  # Custom interval (5 minutes)
  python streaming_collector_sqlite.py --continuous --interval 300
        """
    )
    
    # Database configuration
    parser.add_argument('--database', type=str, default='weather_stream.db', help='SQLite database file path')
    
    # Data source
    parser.add_argument(
        '--url',
        type=str,
        default='https://stratus.meteo.noa.gr/data/stations/latestValues_Datagems.geojson',
        help='GeoJSON data feed URL'
    )
    
    # Collection interval
    parser.add_argument('--interval', type=int, default=600, help='Collection interval in seconds')
    
    # Mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--once', action='store_true', help='Run a single collection cycle')
    mode_group.add_argument('--continuous', action='store_true', help='Continuously collect data')
    
    parser.add_argument('--stats', action='store_true', help='Display database statistics')
    
    args = parser.parse_args()
    
    # Initialize DB
    logger.info("Connecting to SQLite...")
    db = SQLiteDatabase(args.database)
    
    # Optional stats
    if args.stats:
        stats = db.get_stats()
        print("SQLite Database Statistics")
        print(f"Database file: {args.database}")
        print(f"Database size: {stats.get('database_size_mb', 'N/A')}")
        print(f"Stations: {stats.get('stations_count', 0)}")
        print(f"Observations: {stats.get('observations_count', 0)}")
        print(f"Earliest data: {stats.get('earliest_data', 'N/A')}")
        print(f"Latest data: {stats.get('latest_data', 'N/A')}")
        
        if stats.get('last_collection'):
            print(f"\nLast collection:")
            print(f"  Timestamp: {stats['last_collection']['timestamp']}")
            print(f"  Status: {stats['last_collection']['status']}")
            print(f"  Stations: {stats['last_collection']['stations_count']}")
            print(f"  Observations: {stats['last_collection']['observations_count']}")
        
    
    # Build collector
    collector = StreamingCollector(args.url, db, args.interval)
    
    if args.once:
        # Single run
        success, msg = collector.fetch_and_store()
        if success:
            logger.info("✓ Collection succeeded")
            if args.stats:
                stats = db.get_stats()
                print(f"Total observations after run: {stats.get('observations_count', 0)}")
        else:
            logger.error("✗ Collection failed")
        db.close()
    
    else:  # continuous
        # Continuous mode
        print("SQLite Streaming Data Collector")
        print(f"Source URL : {args.url}")
        print(f"Database   : {args.database}")
        print(f"Interval   : {args.interval} seconds")
        print(f"Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Signal handling
        def signal_handler(sig, frame):
            print("\n\nStopping collector ...")
            collector.stop()
            db.close()
            print("Exited cleanly")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start collection
        collector.start()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            collector.stop()
            db.close()


if __name__ == '__main__':
    main()

