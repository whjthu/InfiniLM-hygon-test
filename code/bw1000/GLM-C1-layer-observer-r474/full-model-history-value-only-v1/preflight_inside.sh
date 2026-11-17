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
    *) exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
exec /usr/bin/python3 -B -u "$src/preflight.py" "${1:?missing original scope}"
