#!/usr/bin/env python3
"""Run the whole acceptance test suite and write one summary.

Order matters: the catalogue check needs no service, the functional test
needs a live service, and the latency test is the long one.

Usage:
    python acceptance_tests/run_all.py --manifest acceptance_tests/dataset/manifest.jsonl
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import write_json

HERE = Path(__file__).resolve().parent


def run(script: str, args: list[str]) -> tuple[int, str]:
    command = [sys.executable, str(HERE / script), *args]
    print(f"\n=== {script} ===", flush=True)
    process = subprocess.run(command, cwd=HERE.parent, text=True, capture_output=True)
    output = (process.stdout or "") + (process.stderr or "")
    print(output[-4000:])
    return process.returncode, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("acceptance_tests/dataset/manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("acceptance_tests/results"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    codes: dict[str, int] = {}

    codes["catalog"], _ = run("check_regulation_catalog.py", ["--out", str(args.out / "catalog")])

    common_args = ["--manifest", str(args.manifest)]
    if args.base_url:
        common_args += ["--base-url", args.base_url]
    codes["function"], _ = run("test_function_api.py", common_args + ["--out", str(args.out / "function")])

    if not args.skip_latency:
        latency_args = common_args + ["--out", str(args.out / "latency")]
        if args.limit:
            latency_args += ["--limit", str(args.limit)]
        codes["latency"], _ = run("test_latency.py", latency_args)

    summary = {
        "exit_codes": codes,
        "all_passed": all(code == 0 for code in codes.values()),
        "elapsed_seconds": round(time.time() - started, 1),
        "results_dir": str(args.out),
    }
    write_json(args.out / "acceptance_summary.json", summary)
    print("\n=== 汇总 ===")
    for name, code in codes.items():
        print(f"  {name:<10} {'通过' if code == 0 else '不通过'}")
    print(f"  用时 {summary['elapsed_seconds']}s，记录目录 {args.out}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
