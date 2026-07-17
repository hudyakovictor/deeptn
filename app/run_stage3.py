#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from app.stage3 import Stage3Config,Stage3Engine
def main():
 p=argparse.ArgumentParser(description='DEEPUTIN app6 stage 3 report');p.add_argument('--analysis',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--overwrite',action='store_true');a=p.parse_args();Stage3Engine(Stage3Config(a.analysis.resolve(),a.output.resolve(),a.overwrite)).run();return 0
if __name__=='__main__':raise SystemExit(main())
