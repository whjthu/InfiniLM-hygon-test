#!/bin/bash
set -euo pipefail
ulimit -c 0
python /refactor/incoming/Kimi-live-taps-r9046/build_live_taps_cpu_r9046.py
