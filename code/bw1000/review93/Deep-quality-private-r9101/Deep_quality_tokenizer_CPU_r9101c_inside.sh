#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
python -B -u /refactor/incoming/prepare_Deep_quality_tokenizer_r9101c.py
