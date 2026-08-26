"""Resumable, byte-checked downloader for the public IMS bearing archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


DEFAULT_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
DEFAULT_SIZE = 1_075_597_174
DEFAULT_ETAG = '"a201e36fe558f3f50509701b6d573532-63"'


def _run_part(url: str, path: Path, start: int, end: int, retries: int) -> None:
    expected = end - start + 1
    if path.is_file() and path.stat().st_size == expected:
        return
    path.unlink(missing_ok=True)
    command = [
        "curl", "-L", "--fail", "--retry", str(retries), "--retry-all-errors",
        "--retry-delay", "5", "--connect-timeout", "30", "--speed-time", "60",
        "--speed-limit", "1024", "--range", f"{start}-{end}", "-o", str(path), url,
    ]
    subprocess.run(command, check=True)
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"Range size mismatch for {path.name}: {actual} != {expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("data/processed/ims_bearing.zip"))
    parser.add_argument("--chunk-size", type=int, default=64 << 20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--total-size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--expected-etag", default=DEFAULT_ETAG)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = output.parent / (output.stem + "_chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ranges = []
    start = 0
    index = 0
    while start < args.total_size:
        end = min(args.total_size - 1, start + args.chunk_size - 1)
        ranges.append((index, start, end))
        start = end + 1
        index += 1

    from concurrent.futures import ThreadPoolExecutor

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(_run_part, args.url, chunk_dir / f"part_{i:04d}.bin", start, end, args.retries)
            for i, start, end in ranges
        ]
        for future in futures:
            future.result()

    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with partial.open("wb") as destination:
        for index, start, end in ranges:
            part = chunk_dir / f"part_{index:04d}.bin"
            expected = end - start + 1
            actual = part.stat().st_size
            if actual != expected:
                raise RuntimeError(f"Missing or truncated part {part}: {actual} != {expected}")
            with part.open("rb") as source:
                while block := source.read(1 << 20):
                    destination.write(block)
    if partial.stat().st_size != args.total_size:
        raise RuntimeError(f"Assembled size mismatch: {partial.stat().st_size} != {args.total_size}")
    digest = _sha256(partial)
    os.replace(partial, output)
    manifest = {
        "url": args.url,
        "expected_etag": args.expected_etag,
        "total_size": args.total_size,
        "sha256": digest,
        "chunk_size": args.chunk_size,
        "chunks": len(ranges),
        "elapsed_sec": time.time() - started,
        "complete": True,
    }
    manifest_path = args.manifest or output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
