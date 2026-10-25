#!/bin/bash
set -euo pipefail
timeout --kill-after=5 150 python -B -u /refactor/incoming/Kimi-MoE-lazy-profile-r9125b/run_prewarm_CPU.py
