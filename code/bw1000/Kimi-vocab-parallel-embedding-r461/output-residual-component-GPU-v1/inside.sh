#!/bin/bash
set -euo pipefail
ulimit -c 0
test "$SLURM_JOB_ID" = 1103606
test "$(hostname -s)" = f13r1n01
source /opt/dtk/env.sh
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL LD_AUDIT
case "${SLURM_STEP_GPUS:?one assigned Slurm device required}" in
  [0-7]) export HIP_VISIBLE_DEVICES="$SLURM_STEP_GPUS";;
  *) exit 2;;
esac
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export INFINILM_DENSE_PREFILL_F32_R2475=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/old/offline-deps
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
test -f "$ROCBLAS_TENSILE_LIBPATH/TensileLibrary_lazy_gfx936.dat"
export LD_LIBRARY_PATH="/refactor/install/hygon-kimi-5f91ad7-r3672/lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
LEFT=$(python3 -B -c 'import os,time;print(max(0,int(float(os.environ["KIMI_RESIDUAL_EPOCH"])+190-time.time())))')
test "$LEFT" -gt 35
exec timeout --kill-after=5s "$LEFT" python3 -B -u /refactor/incoming/Kimi-vocab-parallel-embedding-r461/output-residual-component-GPU-v1/worker.py
