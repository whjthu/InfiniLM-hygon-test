#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
python -B -u /refactor/incoming/build_Deep_packed_varlen_r9101b.py
