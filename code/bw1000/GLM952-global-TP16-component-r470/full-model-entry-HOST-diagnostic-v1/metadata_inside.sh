#!/bin/bash
set -euo pipefail
printf '{"phase":"container_bash_entry"}\n'
printf '{"phase":"source_ENV_before"}\n'
source /refactor/incoming/GLM952-global-TP16-component-r470/full-model-entry-v1/ENV.sh
printf '{"phase":"source_ENV_after"}\n'
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
cd /refactor/incoming/GLM952-global-TP16-component-r470/full-model-entry-v1
printf '{"phase":"Python_before"}\n'
/usr/bin/python3 -B /refactor/incoming/GLM952-global-TP16-component-r470/full-model-entry-HOST-diagnostic-v1/metadata.py
printf '{"phase":"Python_after_ldd_before"}\n'
ldd -r /refactor/candidates/GLM-global-TP16-full-native-r470-CPU-v4/libGLM_global_TP16_full_native_r470.so
printf '{"phase":"ldd_after"}\n'
