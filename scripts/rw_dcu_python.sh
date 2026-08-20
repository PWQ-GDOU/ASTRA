#!/usr/bin/env bash
# Run inside the DTK root filesystem after chroot.
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3}"
source /opt/dtk-26.04/env.sh
exec /usr/local/bin/python3.11 "$@"
