#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
# Set after srun/container setup; scheduler exports cannot undo CPU isolation.
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:${PATH}
SOURCE=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
TORCH_LIB_DIR=$(python -c 'import pathlib,importlib.util;print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib")')
export PYTHONPATH="$SOURCE/python:/old/offline-deps:${PYTHONPATH:-}"
PLUGIN=/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt
export LD_LIBRARY_PATH="/refactor/candidates/kimi-helper-bounded-bf16-r6054:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$SOURCE/python:$SOURCE/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$PLUGIN/rccl-shca-net/lib:$PLUGIN/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD LD_AUDIT
left=$(python -c 'import math, os, time
epoch = float(os.environ["KIMI_TILE_EPOCH"])
deadline = float(os.environ["KIMI_TILE_INNER_DEADLINE"])
now = time.time()
if not (math.isfinite(epoch) and math.isfinite(deadline) and epoch > 0 and deadline == epoch + 260 and epoch - 2 <= now < deadline - 5):
    raise SystemExit("invalid or expired inherited build deadline")
print(deadline - now - 5)')
exec timeout --kill-after=5s "${left}s" python -B /refactor/incoming/Kimi-vocab-parallel-embedding-r461/output-residual-stage-diagnostic-CPU-v1/CPU_build.py
