#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
TORCH_LIB_DIR=$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent/"lib")')
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}" OMP_NUM_THREADS=1 INFINILM_OWNER_CPU_WIDTH_R2392=1
export PYTHONPYCACHEPREFIX=/local/kda-serial-r9020-pycache
python - <<'PY'
import json,pathlib,hashlib,os
assert len(os.sched_getaffinity(0))>=8 and os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']=='1'
r=pathlib.Path('/refactor');m=json.loads((r/'incoming/Kimi-serial-manifest-r9020.json').read_text())
assert hashlib.sha256((r/m['native']).read_bytes()).hexdigest()==m['native_SHA256']
for n,h in m['sources_SHA256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
for n,h in m['runtime_dependencies_SHA256'].items():
 p=(r/'install/hygon-kimi-5f91ad7-r3672/lib'/n)if n!='libinfinilm_runtime.so'else(r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib'/n)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
PY
timeout --kill-after=10 210 python -u /refactor/incoming/check_Kimi_serial_r9020.py --gpu-lib /refactor/candidates/Kimi-KDA-only-serial-r9020/libKimi_KDA_only_serial_r9020.so --root-go --out "/refactor/results/Kimi-KDA-only-serial-r9020-${SLURM_JOB_ID}.${SLURM_STEP_ID}.json"
