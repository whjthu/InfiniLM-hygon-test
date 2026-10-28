#!/usr/bin/env bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
cd /refactor/incoming/Kimi-W13-directdecode-resume-r161a
: "${MOE_OBJECT_SHA_R161A:?exact CPU510 moe.o SHA required}"
: "${MOE_MD_SHA_R161A:?exact CPU510 moe.d SHA required}"
exec python3 -B -u CPU_resume_r161a.py --moe-sha "$MOE_OBJECT_SHA_R161A" --moe-md-sha "$MOE_MD_SHA_R161A"
