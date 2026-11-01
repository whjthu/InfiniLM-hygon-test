#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=/refactor/incoming/Deep-C4-paged-score-r359-source-cpu-v1
python -B -c 'import pathlib,hashlib,json,os;p=pathlib.Path("/refactor/incoming/Deep-C4-paged-score-r359-source-cpu-v1");m=p/"SOURCE_MANIFEST.json";assert hashlib.sha256(m.read_bytes()).hexdigest()==os.environ["DEEP359_SOURCE_SHA"];d=json.loads(m.read_bytes());assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in d["files_SHA256"].items())'
exec timeout --signal=TERM --kill-after=10s 540s python -B "$D/build_source_CPU.py"
