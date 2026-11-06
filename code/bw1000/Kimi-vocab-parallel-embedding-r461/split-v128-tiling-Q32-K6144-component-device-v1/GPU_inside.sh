#!/bin/bash
set -euo pipefail
ulimit -c 0
test "$SLURM_JOB_ID" = 910380
test "$(hostname -s)" = a14r2n08
source /refactor/incoming/Kimi-vocab-parallel-embedding-r461/split-v128-tiling-Q32-K6144-component-device-v1/ENV.sh
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL INFINILM_OWNER_CPU_WIDTH_R2392
case "${SLURM_STEP_GPUS:?Actual one GPU allocation ID required}" in [0-7]) export HIP_VISIBLE_DEVICES="$SLURM_STEP_GPUS";; *) echo 'Unsupported allocated single GPU identity' >&2; exit 2;; esac
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
# Exact full path snapshot actually qualified by CPU1089, including inherited tail.
export LD_LIBRARY_PATH='/refactor/candidates/Kimi-split-V128-tiling-link-CPU-v2/Q32-K6144:/refactor/candidates/kimi-helper-bounded-bf16-r6054:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:/refactor/install/hygon-kimi-5f91ad7-r3672/lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python:/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt/rccl-shca-net/lib:/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib::/.singularity.d/libs'
LEFT=$(python3 -B -c 'import os,time;print(max(0,int(float(os.environ["KIMI_SPLITV_GPU_START"])+560-time.time())))')
test "$LEFT" -gt 30
exec timeout --kill-after=10s "$LEFT" python3 -B -u /refactor/incoming/Kimi-vocab-parallel-embedding-r461/split-v128-tiling-Q32-K6144-component-device-v1/GPU_run.py
