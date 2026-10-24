#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
python /refactor/incoming/build_Kimi_owner_events_r8669zm.py
