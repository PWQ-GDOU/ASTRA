#!/bin/bash
cd /data/disk3/spacecraft_rul
source venv/bin/activate
export PYTHONPATH=/data/disk3/spacecraft_rul
pkill -f hybrid 2>/dev/null
sleep 1
nohup python -u scripts/hybrid.py --device cuda:2 --fd all --epochs 200 > outputs/hybrid.log 2>&1 &
echo "PID=$!"
sleep 3
tail -2 outputs/hybrid.log
