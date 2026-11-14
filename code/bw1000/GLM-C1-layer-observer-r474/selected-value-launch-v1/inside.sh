#!/bin/bash
set -euo pipefail
source /paired-worker/ENV.sh
mkdir -m700 /local/transport-verbs
ln -s /rdma-provider-stack/libibverbs.so.1 /local/transport-verbs/libibverbs.so
export LD_LIBRARY_PATH=/candidate-opt/rccl-shca-net/lib:/candidate-opt/ucx-shca/lib:/local/transport-verbs:/rdma-provider-stack:$LD_LIBRARY_PATH
export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export GLM_GLOBAL_TP16_R470=1 GLM_BOUNDED_ROPE_R4279=1 MKL_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
LEFT=$(/usr/bin/python3 -B -c 'import math,os,time;e=float(os.environ["GLM_SELECTED_EPOCH"]);n=time.time();assert math.isfinite(e) and 0<=n-e<240;print(int(e+248-n))')
exec /usr/bin/timeout --kill-after=8s "$LEFT" /usr/bin/python3 -B -u /paired-worker/component_control.py \
    --node "$SLURM_PROCID" --out "$1" --model /old/models/ZhipuAI-GLM-5.3 \
    --binding /paired-worker/CHECKPOINT_RANGES.json
