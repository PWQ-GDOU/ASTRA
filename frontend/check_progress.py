#!/usr/bin/env python3
"""检查所有计划任务是否已完成。"""
import json, sys
from pathlib import Path

PROGRESS_FILE = Path(__file__).parent / "progress.json"

def main():
    if not PROGRESS_FILE.exists():
        print("no progress file — tasks not started")
        sys.exit(1)

    with open(PROGRESS_FILE) as f:
        p = json.load(f)

    remaining = [tid for tid, t in p["tasks"].items() if t["status"] != "done"]
    if not remaining:
        print("all tasks done")
        print(f"total: {p['total_tasks']}, completed: {p['completed']}, rounds: {p['rounds']}")
        sys.exit(0)

    print(f"{len(remaining)} tasks remaining out of {p['total_tasks']}")
    for tid in sorted(remaining, key=int)[:20]:
        print(f"  task {tid}: {p['tasks'][tid]['status']}")
    if len(remaining) > 20:
        print(f"  ... and {len(remaining) - 20} more")
    sys.exit(1)

if __name__ == "__main__":
    main()
