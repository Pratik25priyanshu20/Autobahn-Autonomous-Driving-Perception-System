#!/usr/bin/env bash
set -euo pipefail

python src/app.py --config configs/system.yaml --input video "$@"
