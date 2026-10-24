#!/bin/bash
set -euo pipefail
ulimit -c 0
test "${KIMI_AUX0_ROOT_GO_R8473:-0}" = 1
test "${SLURM_JOB_ID:-}" = 910374
test "$(hostname -s)" = a02r4n01
test -n "${KIMI_AUX0_WATCH_OWNER_R8473:-}"
test -n "${KIMI_AUX0_READY_RECEIPT_R8473:-}"
python - <<'PY'
import os,json,time,pathlib
q=json.loads(pathlib.Path(os.environ['KIMI_AUX0_READY_RECEIPT_R8473']).read_text())
assert q['job_id']=='910374' and q['node']=='a02r4n01'
assert q['watch_owner']==os.environ['KIMI_AUX0_WATCH_OWNER_R8473']
assert q['fresh_Slurm_no_conflict'] and q['whole_node_KFD_empty'] and q['watch_ready']
assert q['lease_seconds']<=600 and q['issued_unix']<=time.time()<q['expires_unix']<=q['issued_unix']+600
assert time.time()-q['issued_unix']<=60
PY
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX=/local/K7-MoE-r8476-${SLURM_JOB_ID}-${SLURM_STEP_ID}
BASE=/refactor/candidates/kimi-K7-MoE-profile-r8082
ACTUAL=/refactor/candidates/kimi-router-M1-5f91ad7-r4635/InfiniLM
PARENT=/refactor/candidates/kimi-dense0-5f91ad7-r4346
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="/opt/dtk/lib:/refactor/candidates/kimi-joint-expert-5f91ad7-r4620:$PARENT:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}" OMP_NUM_THREADS=1
unset LD_PRELOAD NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
test -f "$ROCBLAS_TENSILE_LIBPATH/TensileLibrary_lazy_gfx936.dat"
export INFINILM_KIMI_SPEC_MATH_R3676=11 INFINILM_KIMI_ROUTER_M1_R4635=1 INFINILM_KIMI_MOE_AUX_M1_R4430=1
export INFINILM_DENSE_PREFILL_F32_R2475=1 KIMI_MOE_LAYERS_R4512='[20]'
set +e
timeout --kill-after=10s 540 python -u /refactor/incoming/probe_Kimi_AUX0_r8476.py
result=$?
printf 'KIMI_AUX0_R8473_NUMERIC_RC=%s\n' "$result"
exit "$result"
