#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:/refactor/install/hygon-deep-5f91ad7-r3672/lib:/refactor/candidates/latest-family-native-r269/Deep/python:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:${LD_LIBRARY_PATH:-}
D=/refactor/incoming/Deep-C4-paged-score-r359-env-check-v3
python -B -c 'import pathlib,hashlib,json,os;p=pathlib.Path("/refactor/incoming/Deep-C4-paged-score-r359-env-check-v3");m=p/"SOURCE_MANIFEST.json";assert hashlib.sha256(m.read_bytes()).hexdigest()==os.environ["DEEP359_ENV_SHA"];d=json.loads(m.read_bytes());assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in d["files_SHA256"].items())'
exec timeout --signal=TERM --kill-after=10s 150s python -B "$D/check_env_CPU.py"
