#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD
r466_self="${BASH_SOURCE[0]}"
if [ "${r466_self##*/}" = job-inside.sh ]; then
  test "${WAVE_INSIDE_R2323:?}" = GLM-W4-remapped-final-slots-r469/deployment-v1/client_inside.sh
  r466_source="/refactor/incoming/${WAVE_INSIDE_R2323}"
  if [ ! -f "$r466_source" ]; then
    r466_source="/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/${WAVE_INSIDE_R2323}"
  fi
  cmp -- "$r466_source" "$r466_self"
  r466_self="$r466_source"
fi
r466_dir="$(CDPATH= cd -- "$(dirname -- "$r466_self")" && pwd -P)"
test -f "$r466_dir/client_dispatch.py"
if [ "${1:-}" = --source-only ]; then
  printf '%s\n' "$r466_dir/client_dispatch.py"
  exit 0
fi
exec python3 -B -u "$r466_dir/client_dispatch.py"
