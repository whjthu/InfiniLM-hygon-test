#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
timeout --kill-after=20s 3500 python -u /refactor/incoming/run_GLM_DP_node_r8868.py
