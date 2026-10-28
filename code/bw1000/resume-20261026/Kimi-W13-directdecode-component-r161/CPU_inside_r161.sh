#!/usr/bin/env bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
cd /refactor/incoming/Kimi-W13-directdecode-component-r161
python3 -B -u CPU_build_r161.py
exec python3 -B -u CPU_ISA_review_r161.py
