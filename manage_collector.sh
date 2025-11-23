#!/bin/bash

# Weather Data Collector Management Script
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
source ~/software/miniconda3/bin/activate datagem

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_help() {
    cat << EOF
========================================
Weather Data Collector Manager
========================================

Usage: $0 <command>

Commands:
  start       Start the data collector
  stop        Stop the data collector
  restart     Restart the data collector
  status      Show running status
  stats       Show database statistics
  logs        Show real-time logs
  test        Test single collection
  
Examples:
  $0 start        # Start collector
  $0 status       # Check status
  $0 logs         # View logs

========================================
EOF
}

check_status() {
    if pgrep -f "streaming_collector_sqlite.py --continuous" > /dev/null; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

start_collector() {
    if check_status; then
        echo -e "${YELLOW}⚠ Collector is already running${NC}"
        echo "PID: $(pgrep -f 'streaming_collector_sqlite.py --continuous' | head -1)"
        return
    fi
    
    echo -e "${GREEN}Starting data collector...${NC}"
    nohup python streaming_collector_sqlite.py --continuous --interval 600 > collector_output.log 2>&1 &
    PID=$!
    sleep 2
    
    if check_status; then
        echo -e "${GREEN}✓ Collector started${NC}"
        echo "PID: $PID"
        echo "Database: weather_stream.db"
        echo "Log: streaming_collector.log"
        echo ""
        echo "View logs: $0 logs"
    else
        echo -e "${RED}✗ Failed to start${NC}"
        cat collector_output.log
    fi
}

stop_collector() {
    if ! check_status; then
        echo -e "${YELLOW}⚠ Collector is not running${NC}"
        return
    fi
    
    echo -e "${YELLOW} Stopping data collector...${NC}"
    pkill -f "streaming_collector_sqlite.py --continuous"
    sleep 2
    
    if ! check_status; then
        echo -e "${GREEN}✓ Collector stopped${NC}"
    else
        echo -e "${RED}✗ Failed to stop, trying force kill...${NC}"
        pkill -9 -f "streaming_collector_sqlite.py"
    fi
}

show_status() {
    echo "========================================"
    echo "Data Collector Status"
    echo "========================================"
    
    if check_status; then
        PID=$(pgrep -f "streaming_collector_sqlite.py --continuous" | head -1)
        echo -e "Status: ${GREEN}✓ Running${NC}"
        echo "PID: $PID"
        echo "Started: $(ps -p $PID -o lstart= 2>/dev/null || echo 'N/A')"
        echo ""
        echo "Recent logs:"
        tail -5 streaming_collector.log 2>/dev/null || echo "No logs available"
    else
        echo -e "Status: ${RED}✗ Not running${NC}"
    fi
    
    echo ""
    echo "========================================"
}

show_stats() {
    echo "========================================"
    echo "Database Statistics"
    echo "========================================"
    python streaming_collector_sqlite.py --once --stats 2>/dev/null || true
}

show_logs() {
    echo "========================================"
    echo "Real-time Logs (Press Ctrl+C to exit)"
    echo "========================================"
    tail -f streaming_collector.log
}

test_collection() {
    echo "========================================"
    echo "Test Single Collection"
    echo "========================================"
    python streaming_collector_sqlite.py --once --stats
}

# Main logic
case "${1:-}" in
    start)
        start_collector
        ;;
    stop)
        stop_collector
        ;;
    restart)
        stop_collector
        sleep 2
        start_collector
        ;;
    status)
        show_status
        ;;
    stats)
        show_stats
        ;;
    logs)
        show_logs
        ;;
    test)
        test_collection
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$1'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

