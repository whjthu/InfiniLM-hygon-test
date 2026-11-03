#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export KIMI410_ORDINARY_RUNTIME=1 KIMI410_NATIVE_SECONDS=5100 GLOBAL_MODEL_MODE_R2354=shca
export BASH_ENV=/refactor/incoming/Kimi-TP8PP4-ordinary-r410/runtime-v2/CPU_bind_hooks.sh
exec timeout --kill-after=5s 270 /bin/bash --noprofile --norc /refactor/incoming/Kimi-TP8PP4-ordinary-r410/runtime-v2/target_inside.sh
