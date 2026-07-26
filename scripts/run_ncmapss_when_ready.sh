#!/usr/bin/env bash
# Auto-extract N-CMAPSS and launch strict benchmark when download completes.
# Usage: bash scripts/run_ncmapss_when_ready.sh [--device cuda:0]
set -euo pipefail

PROJ=/data/disk3/spacecraft_rul
ZIP=$PROJ/data/processed/ncmapss/N-CMAPSS_full.zip
EXTRACT=$PROJ/data/processed/ncmapss
OUT=$PROJ/outputs/ncmapss_strict_v1
DEVICE=${1:-cuda:0}
LOG=/tmp/ncmapss_benchmark.log

echo "[$(date)] Waiting for download to complete..."
while [ ! -f "$ZIP" ] || [ -f "${ZIP}.aria2" ]; do
    sleep 60
    if [ -f "$ZIP" ] && [ ! -f "${ZIP}.aria2" ]; then break; fi
    sz=$(du -sh "$ZIP" 2>/dev/null | cut -f1 || echo '?')
    echo "[$(date)] Download in progress: $sz"
done

echo "[$(date)] Download complete. Extracting..."
cd "$EXTRACT"
unzip -o "$ZIP" -d . 2>&1 | tail -5

# Find and flatten h5 files into the ncmapss directory
find "$EXTRACT" -name "*.h5" | while read f; do
    base=$(basename "$f")
    [ -f "$EXTRACT/$base" ] || mv "$f" "$EXTRACT/$base"
done
echo "[$(date)] HDF5 files:"
ls -lh "$EXTRACT"/*.h5 2>/dev/null || echo "No .h5 files found"

echo "[$(date)] Starting N-CMAPSS benchmark on $DEVICE..."
cd "$PROJ"
env PYTHONPATH=. CUDA_VISIBLE_DEVICES=${DEVICE##cuda:} ./venv/bin/python \
    scripts/exp_ncmapss_strict.py \
    --data-dir data/processed/ncmapss \
    --datasets DS01 DS02 DS03 DS04 \
    --output "$OUT" \
    --device "$DEVICE" \
    > "$LOG" 2>&1 &

echo "[$(date)] Benchmark started, PID=$!, log=$LOG"
echo "Monitor: tail -f $LOG"
