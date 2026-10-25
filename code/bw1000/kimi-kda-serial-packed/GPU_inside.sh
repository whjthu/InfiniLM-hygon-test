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
export OMP_NUM_THREADS=1 INFINILM_OWNER_CPU_WIDTH_R2392=1
python - <<'PY'
import json,pathlib,hashlib,os
assert len(os.sched_getaffinity(0))>=8 and os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']=='1'
r=pathlib.Path('/refactor');m=json.loads((r/'incoming/Kimi-serial-packed-manifest-r9025.json').read_text())
assert hashlib.sha256((r/m['native']).read_bytes()).hexdigest()==m['native_SHA256']
for n,h in m['sources_SHA256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
for n,h in m['runtime_dependencies_SHA256'].items():
 p=(r/'install/hygon-kimi-5f91ad7-r3672/lib'/n)if n!='libinfinilm_runtime.so'else(r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib'/n)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
PY
timeout --kill-after=10 210 python -u /refactor/incoming/check_Kimi_serial_r9020.py --gpu-lib /refactor/candidates/Kimi-KDA-only-serial-packed-r9025/libKimi_KDA_only_serial_packed_r9025.so --root-go --out "/refactor/results/Kimi-KDA-only-serial-packed-r9025-${SLURM_JOB_ID}.${SLURM_STEP_ID}.json"
