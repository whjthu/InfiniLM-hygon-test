#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
TORCH_LIB_DIR=$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent/"lib")')
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}" OMP_NUM_THREADS=1 INFINILM_OWNER_CPU_WIDTH_R2392=4
export PYTHONPYCACHEPREFIX=/local/kda-r9004-pycache
python - <<'PY'
import json,pathlib,hashlib
r=pathlib.Path('/refactor');pin=json.loads((r/'incoming/Kimi-KDA-component-dependencies-r9004.json').read_text())
for n,h in pin['files_SHA256'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
p=r/'candidates/Kimi-current-KDA-r9004/libKimi_current_KDA_r9004.so'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='e5a074b402bbb518742de702a07e9d0c2e2ce04a370c610a12f944edb3ea45ea'
p=r/'incoming/check_Kimi_current_KDA_lifecycle_r9004c.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='2581dbd9b18bcd8356c89e56cebbca173385d699569f210a52b5a428e17b1c69'
PY
timeout --kill-after=10 300 python -u /refactor/incoming/check_Kimi_current_KDA_lifecycle_r9004c.py --gpu-lib /refactor/candidates/Kimi-current-KDA-r9004/libKimi_current_KDA_r9004.so --device 0 --root-go --out "/refactor/results/Kimi-current-KDA-r9004-${SLURM_JOB_ID}.${SLURM_STEP_ID}.json"
