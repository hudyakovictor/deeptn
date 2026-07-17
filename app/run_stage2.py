#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from app.stage2 import Stage2Config,Stage2Engine

def main():
 p=argparse.ArgumentParser(description='DEEPUTIN app6 stage 2')
 p.add_argument('--stage1',required=True,type=Path)
 p.add_argument('--calibration',required=True,type=Path)
 p.add_argument('--output',required=True,type=Path)
 p.add_argument('--overwrite',action='store_true')
 p.add_argument('--lead-archive',type=Path,help='Prior final_inference/add archive; used for coverage auditing only')
 a=p.parse_args()
 Stage2Engine(Stage2Config(a.stage1.resolve(),a.calibration.resolve(),a.output.resolve(),a.overwrite,lead_archive=a.lead_archive.resolve() if a.lead_archive else None)).run()
 return 0
if __name__=='__main__':raise SystemExit(main())
