#!/bin/bash
# Quick launch: kill old and start V2 training
cd /data/disk3/spacecraft_rul
source venv/bin/activate
export PYTHONPATH=/data/disk3/spacecraft_rul
pkill -f train 2>/dev/null
sleep 1
nohup python scripts/train_v2.py --gpu 1 --seed 42 --epochs 200 > outputs/t1.log 2>&1 &
nohup python scripts/train_v2.py --gpu 2 --seed 123 --epochs 200 > outputs/t2.log 2>&1 &
nohup python scripts/train_v2.py --gpu 3 --seed 456 --epochs 200 > outputs/t3.log 2>&1 &
echo "Launched V2 training on GPU 1-3"
sleep 8
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
