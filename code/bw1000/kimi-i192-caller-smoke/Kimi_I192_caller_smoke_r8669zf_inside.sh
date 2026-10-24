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
export INFINILM_KIMI_MLA_INPUT_R4465=1
unset INFINILM_KIMI_KDA_GATE_M1_R4576
export INFINILM_KIMI_KDA_GATE_M1_R4571=3
export INFINILM_KIMI_ROUTER_M1_R4635=1
export INFINILM_KIMI_MOE_AUX_M1_R4430=1
export INFINILM_KIMI_SPEC_R3632=1 INFINILM_KIMI_KDA_INPUT_M1_R4305=1 INFINILM_KIMI_DENSE0_M1_R4346=1
unset INFINILM_KIMI_SPEC_R3632 INFINILM_KIMI_KDA_INPUT_M1_R4305 INFINILM_KIMI_DENSE0_M1_R4346 INFINILM_KIMI_MOE_AUX_M1_R4430 INFINILM_KIMI_MLA_INPUT_R4465 INFINILM_KIMI_KDA_GATE_M1_R4571 INFINILM_KIMI_ROUTER_M1_R4635
export INFINILM_KIMI_VOCAB_HEAD_R8341=0
export INFINILM_KIMI_MOE_WORKSPACE_R4665=1 INFINILM_KIMI_MOE_EXECUTION_R4665=0 INFINILM_KIMI_N4_TILE_R8167=0
export INFINILM_KIMI_CCL512_R8603=0
export INFINILM_KIMI_PREFILL512_R8562=0 INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562=0
export INFINILM_KIMI_I192_GROUPED_R8669Y=0
export INFINILM_KIMI_I192_COMPACT_R8669ZC=1
timeout --kill-after=10s 240 python -u /refactor/incoming/Kimi_I192_caller_smoke_r8669zf.py
