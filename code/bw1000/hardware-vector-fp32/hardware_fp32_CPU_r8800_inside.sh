#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
python -u /refactor/incoming/build_hardware_fp32_fma_r8800.py
