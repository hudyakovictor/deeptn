"""Run ITER1..ITER9 unit suites."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SUITES = [
    "tests/test_iter1_sensor.py",
    "tests/test_iter2_visibility_uv.py",
    "tests/test_iter3_alignment_zones.py",
    "tests/test_iter4_extraction.py",
    "tests/test_iter5_calibration.py",
    "tests/test_iter6_compare.py",
    "tests/test_iter7_texture.py",
    "tests/test_iter8_verdict.py",
    "tests/test_iter9_report_acceptance.py",
]

def main() -> int:
    failed = []
    for rel in SUITES:
        path = ROOT / rel
        print("=" * 60)
        print("RUN", rel)
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as e:
            if e.code not in (0, None):
                failed.append(rel)
        except Exception as e:
            print("FAIL", rel, e)
            failed.append(rel)
    print("=" * 60)
    if failed:
        print("FAILED:", failed)
        return 1
    print("ALL ITER1-9 SUITES PASSED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
