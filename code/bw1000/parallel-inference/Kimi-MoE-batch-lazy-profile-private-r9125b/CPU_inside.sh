#!/bin/bash
set -euo pipefail
export PYTHONPATH=/old/offline-deps:${PYTHONPATH:-}
python -B -u /refactor/incoming/Kimi-MoE-lazy-profile-r9125b/CPU_contract.py
