#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 -m unittest discover -s app/tests -p 'test_*.py' -v
python3 -m compileall -q app
