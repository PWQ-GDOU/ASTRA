#!/bin/bash
# Launch SHRDL pretraining on GPU 1-3 in parallel
# Usage: bash /data/disk3/spacecraft_rul/scripts/launch_training.sh

set -e

PROJECT_DIR="/data/disk3/spacecraft_rul"
cd "$PROJECT_DIR"
source venv/bin/activate
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "============================================"
echo "  Spacecraft RUL — SHRDL Pretraining"
echo "  GPUs: 1, 2, 3 (GPU 0 reserved)"
echo "  Time: $(date)"
echo "============================================"

# Kill any existing training
pkill -f train_server.py 2>/dev/null || true
sleep 2

# Launch GPU 1
echo "[GPU1] Starting (seed=42)..."
nohup python scripts/train_server.py --gpu 1 --seed 42 --epochs 200 --batch_size 128 \
    > outputs/train_gpu1.log 2>&1 &
echo "  PID=$!"

# Launch GPU 2
echo "[GPU2] Starting (seed=123)..."
nohup python scripts/train_server.py --gpu 2 --seed 123 --epochs 200 --batch_size 128 \
    > outputs/train_gpu2.log 2>&1 &
echo "  PID=$!"

# Launch GPU 3
echo "[GPU3] Starting (seed=456)..."
nohup python scripts/train_server.py --gpu 3 --seed 456 --epochs 200 --batch_size 128 \
    > outputs/train_gpu3.log 2>&1 &
echo "  PID=$!"

sleep 10

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader

echo ""
echo "=== Training Logs ==="
for gpu in 1 2 3; do
    echo "--- GPU$gpu (tail -3) ---"
    tail -3 outputs/train_gpu${gpu}.log 2>/dev/null || echo "  (waiting...)"
done

echo ""
echo "=== Monitor ==="
echo "  watch -n 1 nvidia-smi"
echo "  tail -f outputs/train_gpu1.log"
echo "  tail -f outputs/train_gpu2.log"
echo "  tail -f outputs/train_gpu3.log"
