#!/bin/bash
set -euo pipefail
timeout --kill-after=5 150 python -B -u /refactor/incoming/Kimi-MoE-prewarm-r9125/run_prewarm_CPU_r9125.py
