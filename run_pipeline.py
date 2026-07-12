#!/usr/bin/env python3
"""
Run DEEPUTIN pipeline on 3 SD card datasets sequentially.
Results are saved to /Volumes/SDCARD/test_photos/results/<dataset>/

Usage:
  python run_pipeline.py              # dry run (limit=3, first dataset only)
  python run_pipeline.py --full       # full run on all 3 datasets
"""
import subprocess
import os
import sys
import time
from pathlib import Path

PYTHON = "/opt/homebrew/Caskroom/miniconda/base/envs/deeputin/bin/python"
DEEPUTIN_ROOT = Path("/Users/victorkhudyakov/deeputin")
RESULTS = Path("/Volumes/SDCARD/test_photos/results")

DATASETS = [
    {
        "name": "put",
        "input": "/Volumes/SDCARD/test_photos/put",
    },
    {
        "name": "udmurt",
        "input": "/Volumes/SDCARD/test_photos/udmurt",
    },
    {
        "name": "vas",
        "input": "/Volumes/SDCARD/test_photos/vas",
    },
]


def run_dataset(cfg: dict, limit: int | None = None) -> bool:
    name = cfg["name"]
    input_dir = cfg["input"]
    data_root = RESULTS / name
    main_output = data_root / "storage" / "main"
    cal_output = data_root / "storage" / "calibration"

    main_output.mkdir(parents=True, exist_ok=True)
    cal_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  DATASET: {name}")
    print(f"  Input:   {input_dir}")
    print(f"  Output:  {main_output}")
    print(f"{'='*70}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = RESULTS / f"pipeline_{name}_{ts}.log"

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "DPTN_DATA_ROOT": str(data_root),
        "DUTIN_3DDFA_PATH": str(DEEPUTIN_ROOT / "core" / "3ddfa_v3"),
        "PYTHONPATH": str(DEEPUTIN_ROOT),
    }

    cmd = [
        str(PYTHON), "-u", "-m", "project.run",
        "--stages", "s1", "s2", "s3", "s4", "s5",
        "--input-main", input_dir,
        "--output-main", str(main_output),
        "--input-calibration", str(main_output),
        "--output-calibration", str(cal_output),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"\nStarting at {time.strftime('%H:%M:%S')}...")
    print(f"Log: {log_file}")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_file, "w")

    proc = subprocess.Popen(
        cmd,
        cwd=str(DEEPUTIN_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    last_lines = []
    MAX_TAIL = 80
    for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace")
        log_fh.write(line)
        log_fh.flush()
        print(line, end="")
        last_lines.append(line)
        if len(last_lines) > MAX_TAIL:
            last_lines.pop(0)

    proc.wait()
    log_fh.close()

    ok = proc.returncode == 0
    print(f"\n{'OK' if ok else 'FAILED'} (exit {proc.returncode})")
    return ok


def main():
    full = "--full" in sys.argv
    RESULTS.mkdir(parents=True, exist_ok=True)

    ts_total = time.time()
    results = {}

    if not full:
        DRY_RUN_LIMIT = 3
        print(f"\n*** DRY RUN: testing with limit={DRY_RUN_LIMIT} on first dataset ***")
        print(f"    Run with --full to process all datasets without limit.\n")
        t0 = time.time()
        cfg = DATASETS[0]
        ok = run_dataset(cfg, limit=DRY_RUN_LIMIT)
        results[cfg["name"]] = {"ok": ok, "elapsed": time.time() - t0}

        print(f"\n{'='*70}")
        print(f"  DRY RUN RESULT: {'OK' if ok else 'FAILED'}")
        print(f"  Time: {int(results[cfg['name']]['elapsed']//60)}m{int(results[cfg['name']]['elapsed']%60):02d}s")
        print(f"{'='*70}")
    else:
        print(f"\n*** FULL RUN: processing all {len(DATASETS)} datasets ***\n")
        for cfg in DATASETS:
            t0 = time.time()
            ok = run_dataset(cfg)
            results[cfg["name"]] = {"ok": ok, "elapsed": time.time() - t0}

        total_elapsed = time.time() - ts_total
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        for name, r in results.items():
            status = "OK" if r["ok"] else "FAILED"
            mins = int(r["elapsed"] // 60)
            secs = int(r["elapsed"] % 60)
            print(f"  {name:12s}  {status:8s}  {mins}m{secs:02d}s")
        t_mins = int(total_elapsed // 60)
        t_secs = int(total_elapsed % 60)
        all_ok = all(r["ok"] for r in results.values())
        print(f"\n  Total: {t_mins}m{t_secs:02d}s  {'ALL OK' if all_ok else 'SOME FAILED'}")
        print(f"  Results: {RESULTS}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()