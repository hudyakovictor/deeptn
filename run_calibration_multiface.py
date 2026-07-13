#!/usr/bin/env python3
"""
Run calibration pipeline on MultiFace dataset.

Usage:
    python run_calibration_multiface.py \
        --multiface-dir /Volumes/SDCARD/calibration_datasets/002643814 \
        --output-dir /Volumes/SDCARD/calibration_datasets/002643814_output
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run calibration pipeline on MultiFace dataset")
    parser.add_argument("--multiface-dir", required=True,
                        help="Path to MultiFace calibration dataset on SD card")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for calibration results")
    parser.add_argument("--stages", nargs="*", default=["s1", "s2"],
                        help="Pipeline stages to run (default: s1 s2)")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--limit", type=int, default=None, help="Limit photos per dataset (for testing)")
    return parser.parse_args()


def verify_paths(multiface_dir, output_dir):
    """Verify input and output paths."""
    mf = Path(multiface_dir)
    out = Path(output_dir)
    
    if not mf.exists():
        print(f"Error: MultiFace directory does not exist: {mf}")
        return False
    
    # Check for images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [f for f in mf.rglob("*") if f.suffix.lower() in exts]
    if not images:
        print(f"Warning: No images found in {mf}")
    else:
        print(f"Found {len(images)} images in MultiFace dataset")
    
    out.mkdir(parents=True, exist_ok=True)
    return True


def run_pipeline(args):
    """Run the deeputin pipeline."""
    cmd = [
        sys.executable, "-m", "project.run",
        "--input-calibration", str(args.multiface_dir),
        "--output-calibration", str(args.output_dir),
        "--stages", *args.stages,
    ]
    
    if args.config:
        cmd.extend(["--config", args.config])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MultiFace Calibration Pipeline")
    print("=" * 60)
    print(f"Input:  {args.multiface_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Stages: {', '.join(args.stages)}")
    print("=" * 60)
    
    if not verify_paths(args.multiface_dir, args.output_dir):
        return 1
    
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())