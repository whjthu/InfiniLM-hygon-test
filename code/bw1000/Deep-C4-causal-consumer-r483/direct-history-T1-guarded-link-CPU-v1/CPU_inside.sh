#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:/refactor/install/hygon-deep-5f91ad7-r3672/lib:/refactor/candidates/latest-family-native-r269/Deep/python:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD LD_AUDIT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PYTHONPATH HIP_CLANG_PATH HIP_PLATFORM HIP_COMPILER
remaining=$(/usr/bin/python3 -I -S -B -c 'import os,time,math; x=min(270.0,float(os.environ["TUNING_CPU_DEADLINE"])-time.time()-7.0); assert math.isfinite(x) and x>2; print(x)')
exec /usr/bin/timeout --kill-after=5s "${remaining}s" /usr/bin/python3 -I -S -B -u - <<'PY'
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import sys

package = Path('/refactor/incoming/DeepC4-T1-history-guarded-probe-link-v1')
raw = (package / 'SOURCE_FILES.json').read_bytes()
if sha256(raw).hexdigest() != os.environ['TUNING_CPU_MANIFEST_SHA']:
    raise ValueError('original staged manifest changed')
manifest = json.loads(raw)
for name, pin in manifest['files'].items():
    path = package / name
    if not path.resolve().is_relative_to(package) or sha256(path.read_bytes()).hexdigest() != pin['SHA256']:
        raise ValueError('original staged source changed: ' + name)
tools = package / 'repo' / 'tools'
sys.path.insert(0, str(tools))
runpy.run_path(str(tools / 'run_deep_c4_direct_history_link.py'), run_name='__main__')
PY
