#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1
unset USE_TORCH USE_TF USE_FLAX BASH_ENV PYTHONSTARTUP PYTHONINSPECT
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:$SDK/lib:/refactor/candidates/latest-family-native-r269/Deep/python:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/refactor/incoming/Deep-fixed-fullpre-experimental-profile-r454/source-v1:/refactor/incoming/public-Deep-native-provider-r427/source-v3:/refactor/candidates/latest-family-native-r269/Deep/python:/old/offline-deps"
unset INFINILM_DEEP_LAYER_TRACE_R4026 NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS INFINILM_DEEP_QUALITY_PREFLIGHT_ONLY_R9101 INFINILM_DEEP_POSITION_PROOF_R8726
export OMP_NUM_THREADS=1 INFINILM_OWNER_CPU_WIDTH_R2392=4
export INFINILM_DEEP_MOE_TAIL512_R7211=0 INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750=0

test "${SLURM_JOB_ID:?}" = 1087656
test "${SLURM_CPUS_PER_TASK:?}" = 64
test "$(hostname -s)" = f12r3n12
deadline="${1:?original deadline epoch required}"
expected_sha="${2:?exact probe SHA required}"
left=$(python3 -B -c '
import math, sys, time
remaining = float(sys.argv[1]) - time.time()
if not math.isfinite(remaining) or not 130 <= remaining <= 1800:
    raise SystemExit("invalid original deadline")
print(int(remaining)-8)
' "$deadline")
exec timeout --kill-after=5s "$left" python3 -B -u -c '
import hashlib, re, sys
expected_sha, deadline = sys.argv[1:]
code = sys.stdin.buffer.read(32769)
if (not re.fullmatch(r"[a-f0-9]{64}", expected_sha)
        or not 0 < len(code) <= 32768
        or b"\r" in code
        or hashlib.sha256(code).hexdigest() != expected_sha):
    raise SystemExit("invalid frozen probe bytes")
sys.argv = ["load_probe.py", "--deadline-epoch", deadline]
exec(compile(code, "load_probe.py", "exec"), {"__name__": "__main__", "__file__": "load_probe.py"})
' "$expected_sha" "$deadline"
