#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONFAULTHANDLER=1
export PYTHONPYCACHEPREFIX="/local/glm-trace8692-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="/refactor/candidates/glm-hybrid-decode-r5386:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/refactor/candidates/glm-grouped-native-model-r5117/helpers:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export GLM_PROFILE_SESSION_R8692="glm8692-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
D="/local/glm-spawn-trace-r8692-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
mkdir -m700 "$D"
timeout 10s /opt/dtk/bin/hipprof -h >"$D/help.txt" 2>&1
python3 -c 'import pathlib,sys; t=pathlib.Path(sys.argv[1]).read_text();required=["--hip-trace","--trace-off","--follow-fork","--show-pid","--session","--output-type"];assert all(x in t for x in required),"PARK_current_CLI_mismatch_NO_GPU_payload"' "$D/help.txt"
python3 /refactor/incoming/run_GLM_spawn_trace_bounded_r8692.py "$D" /opt/dtk/bin/hipprof --hip-trace --trace-off --follow-fork --show-pid --session "$GLM_PROFILE_SESSION_R8692" --output-type 0 -d "$D" -o "$D/trace" python -u /refactor/incoming/GLM_spawn_trace_tiny_r8692.py
python3 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); size=sum(x.stat().st_size for x in p.rglob("*")if x.is_file());assert size<=256*1024**2,size;print("TRACE_RAW_BYTES",size)' "$D"
A="/refactor/results/GLM-spawn-trace-r8692-${SLURM_JOB_ID}.${SLURM_STEP_ID}"
mkdir -m700 "$A"
cp -r "$D"/. "$A/"
