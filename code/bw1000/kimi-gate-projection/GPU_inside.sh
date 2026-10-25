#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
test -f "$ROCBLAS_TENSILE_LIBPATH/TensileLibrary_lazy_gfx936.dat"
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:${PATH}
SOURCE=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export PYTHONPATH="$SOURCE/python:/old/offline-deps:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$SOURCE/python:$SOURCE/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/refactor/candidates/kimi-helper-bounded-bf16-r6054:$LD_LIBRARY_PATH"
export INFINILM_KIMI_PACKED_W2_R6122=0 INFINILM_KIMI_PACKED_AUTO_R6136=0
export INFINILM_KIMI_METADATA_PACK_R5858=0 INFINILM_KIMI_BOUNDED_BF16_R6054=0 INFINILM_KIMI_STATIC_DIV_R6020=0
unset INFINILM_KIMI_COMPACT_CACHE_R4974
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export OMP_NUM_THREADS=1 INFINILM_OWNER_CPU_WIDTH_R2392=1 INFINILM_DENSE_PREFILL_F32_R2475=1
python - <<'PY'
import json,pathlib,hashlib,os
r=pathlib.Path("/refactor");m=json.loads((r/"incoming/Kimi-gate-manifest-r9032.json").read_text());assert len(os.sched_getaffinity(0))>=8 and os.environ["INFINILM_OWNER_CPU_WIDTH_R2392"]=="1"
for n,h in m["sources_SHA256"].items():assert hashlib.sha256((r/"incoming"/n).read_bytes()).hexdigest()==h,n
assert hashlib.sha256(pathlib.Path(m["native"]).read_bytes()).hexdigest()==m["native_SHA256"]
PY
timeout --kill-after=10 210 python -u /refactor/incoming/Kimi_gate_driver_r9032.py
