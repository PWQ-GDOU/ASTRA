#!/bin/bash
cd /data/disk3/spacecraft_rul
source venv/bin/activate
export PYTHONPATH=/data/disk3/spacecraft_rul
pkill -f shrdl 2>/dev/null
sleep 2
nohup python -u scripts/shrdl_v4.py --device cuda:1 --fd FD002 --epochs 1000 --lr 5e-4 --wd 3e-3 --alpha 0.3 > outputs/long_fd002.log 2>&1 &
nohup python -u scripts/shrdl_v4.py --device cuda:2 --fd FD004 --epochs 1000 --lr 5e-4 --wd 3e-3 --alpha 0.3 > outputs/long_fd004.log 2>&1 &
nohup python -u scripts/shrdl_v4.py --device cuda:3 --fd FD001 --epochs 1000 --lr 5e-4 --wd 3e-3 --alpha 0.3 > outputs/long_fd001.log 2>&1 &
echo "Launched 1000-epoch training on GPU1(GPU2(FD004) GPU3(FD001)"
sleep 8
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
