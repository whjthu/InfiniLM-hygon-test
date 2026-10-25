#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
TORCH_LIB_DIR=$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent/"lib")')
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COMMON/python:/refactor/incoming/Kimi-batch-trace-fixture-r9122b:/old/offline-deps:${PYTHONPATH:-}" OMP_NUM_THREADS=1 INFINILM_OWNER_CPU_WIDTH_R2392=1
export PYTHONPYCACHEPREFIX=/local/taps-r9023-pycache
python - <<'CHECK'
import importlib.util,json,pathlib,torch
s=importlib.util.find_spec('infinilm');assert s and s.origin and '/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/'in s.origin
assert not torch.cuda.is_initialized()
q={'passed':True,'find_spec_origin':s.origin,'GPU_initialized':False,'same_GPU_shell_environment':True};pathlib.Path('/refactor/results/Kimi-batch-trace-pathcheck-r9122c.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
CHECK
