#!/bin/bash
set -euo pipefail
ulimit -c 0
src="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$src/ENV.sh"
aliases=/local/transport-verbs
mkdir -m 700 "$aliases"
ln -s /rdma-provider-stack/libibverbs.so.1 "$aliases/libibverbs.so"
ln -s /rdma-provider-stack/libibverbs.so.1 "$aliases/libibverbs.so.1"
export LD_LIBRARY_PATH=/candidate-opt/rccl-shca-net/lib:/candidate-opt/ucx-shca/lib:"$aliases":/rdma-provider-stack:$LD_LIBRARY_PATH
export RDMAV_DRIVERS=/rdma-provider-stack/libshca UCX_MODULE_DIR=/candidate-opt/ucx-shca/lib/ucx
export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
case "$GLM474_TRANSPORT" in
    socket) export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 ;;
    shca) export NCCL_NET=IBext_v8 NCCL_NET_PLUGIN=shca NCCL_IB_DISABLE=0 ;;
    *) exit 2 ;;
esac
export GLM_C1_LAYER_OBSERVER_R474=1
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL INFINILM_OWNER_CPU_WIDTH_R2392
export GLM_GLOBAL_TP16_R470=1 GLM_GLOBAL_TP16_MODEL_R470=1 GLM_V_ON_DEMAND_R403=1 GLM_BOUNDED_ROPE_R4279=1 GLM_SHARED_CONTROL_R4279=1
export GLM_VOCAB_PARALLEL_EMBEDDING_R471=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
LEFT=$(PYTHONPATH="$src" /usr/bin/python3 -B -c 'import os;from budget import INSIDE_SECONDS,remaining;print(max(0,int(remaining(float(os.environ["GLM470_EPOCH"]),INSIDE_SECONDS))))')
test "$LEFT" -gt 60
exec /usr/bin/timeout --kill-after=15s "$LEFT" /usr/bin/python3 -B -u "$src/control.py" worker
