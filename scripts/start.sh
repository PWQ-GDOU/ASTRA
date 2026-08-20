#!/bin/bash
# Launch SHRDL pretraining on GPU 1-3
cd /data/disk3/spacecraft_rul
source venv/bin/activate
export PYTHONPATH=/data/disk3/spacecraft_rul

echo "=== Launching Training $(date) ==="

# GPU 1
python scripts/train_full.py --gpu 1 --seed 42 --epochs 300 --batch_size 128 --seq_len 64 --stride 8 \
    > outputs/train_gpu1.log 2>&1 &
echo "GPU1 PID=$!"

# GPU 2  
python scripts/train_full.py --gpu 2 --seed 123 --epochs 300 --batch_size 128 --seq_len 64 --stride 8 \
    > outputs/train_gpu2.log 2>&1 &
echo "GPU2 PID=$!"

# GPU 3
python scripts/train_full.py --gpu 3 --seed 456 --epochs 300 --batch_size 128 --seq_len 64 --stride 8 \
    > outputs/train_gpu3.log 2>&1 &
echo "GPU3 PID=$!"

echo "All launched. Check with: tail -f outputs/train_gpu1.log"
