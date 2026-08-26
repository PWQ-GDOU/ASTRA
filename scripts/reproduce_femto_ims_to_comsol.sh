#!/usr/bin/env bash
set -euo pipefail

phase="${1:-all}"
run_id="${2:-femto-ims-to-comsol-$(date -u +%Y%m%dT%H%M%SZ)}"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

exec docker compose run --rm --build reproduce \
  python scripts/reproduce_femto_ims_to_comsol.py \
  --phase "$phase" --data-root /workspace/data \
  --output-root "/workspace/outputs/reproducibility/$run_id" --device cpu
