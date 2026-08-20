from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.femto_strict import load_femto_zip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-zip", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but torch.cuda.is_available() is false")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    import scipy
    import sklearn
    import pandas

    data_zip = Path(args.data_zip)
    if not data_zip.is_file():
        raise FileNotFoundError(data_zip)
    with zipfile.ZipFile(data_zip) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"corrupt ZIP member: {bad_member}")

    x = torch.randn(256, 256, device=device)
    y = x @ x
    if not torch.isfinite(y).all().item():
        raise RuntimeError("non-finite preflight matrix result")
    if device.type == "cuda":
        torch.cuda.synchronize()

    series = load_femto_zip(data_zip)
    names = sorted(series)
    if len(series) != 6:
        raise RuntimeError(f"expected 6 FEMTO series, found {len(series)}")

    result = {
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_count": int(torch.cuda.device_count()),
        "torch_threads": int(torch.get_num_threads()),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
        "device": str(device),
        "data_zip_bytes": data_zip.stat().st_size,
        "series": names,
        "matrix_mean": float(y.mean().item()),
    }
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
