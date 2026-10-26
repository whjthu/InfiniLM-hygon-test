#!/bin/bash
set -euo pipefail
# Root may invoke only after publishing the source claim. HOST compile only.
source /opt/dtk/env.sh
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="/refactor/candidates/kimi-helper-bounded-bf16-r6054:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
bash -n /refactor/incoming/Kimi-TP32-I96-MoE-r120/GPU_inside_PROPOSED.sh
python3 -B /refactor/incoming/Kimi-TP32-I96-MoE-r120/CPU_contract_r120.py
python3 -B /refactor/incoming/Kimi-TP32-I96-MoE-r120/CPU_build_r120.py
