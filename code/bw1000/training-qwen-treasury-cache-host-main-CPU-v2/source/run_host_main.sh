#!/bin/bash
set -euo pipefail
umask 077
if [ "$#" -ne 3 ] || [ "$3" != "--host-main-preflight" ]; then
    printf '%s\n' 'Explicit HOST-only optin and original CPU deadline required' >&2
    exit 2
fi
ARM="$1"
case "$ARM" in base|old_domain|domain) ;; *) exit 2 ;; esac
DEADLINE="$2"
ROOT=/public/home/aclsylqidf/hygon-inference-refactor-20261014
RUN="$ROOT/results/training-qwen-treasury-cache-host-main-candidate-v2/arms/$ARM"
SOURCE="$ROOT/results/training-qwen-treasury-cache-host-main-candidate-v2/source"
source /opt/dtk/env.sh >&2
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN SSH_AUTH_SOCK
export TMPDIR="$RUN/cache/tmp" TMP="$RUN/cache/tmp" TEMP="$RUN/cache/tmp"
export XDG_CACHE_HOME="$RUN/cache/xdg" TORCH_EXTENSIONS_DIR="$RUN/cache/torch_extensions"
export TRITON_CACHE_DIR="$RUN/cache/triton" TORCHINDUCTOR_CACHE_DIR="$RUN/cache/torchinductor" AMD_COMGR_CACHE_DIR="$RUN/cache/comgr"
export MIOPEN_USER_DB_PATH="$RUN/cache/miopen" MIOPEN_CUSTOM_CACHE_DIR="$RUN/cache/miopen" MIOPEN_CACHE_DIR="$RUN/cache/miopen"
export HOME="$RUN/cache/home" HF_HOME="$RUN/cache/hf_home" HF_MODULES_CACHE="$RUN/cache/hf_modules"
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SOURCE:/vision:/minicpm-transformers:/swift-runtime/site"
cd -- "$RUN"
python3 -B "$SOURCE/prepare_owned_cache.py" --arm "$ARM" --host-main-preflight
MODEL="$ROOT/results/training-qwen-original-public-model-v1/model"
INPUTS="$ROOT/results/training-qwen-treasury-explicit-dev-CPU-v1/dev"
INVENTORY="$ROOT/results/training-qwen-treasury-explicit-dev-CPU-v1/inventory.json"
ARGS=(--arm "$ARM" --inputs "$INPUTS" --inventory "$INVENTORY" --model "$MODEL" --output "$RUN/host-main" --deadline "$DEADLINE" --host-main-preflight)
case "$ARM" in
    old_domain) ARGS+=(--adapter "$ROOT/results/training-qwen-treasury-sft-v3/continuous/train/checkpoint-5") ;;
    domain) ARGS+=(--adapter "$ROOT/results/training-qwen-treasury-explicit-sft-v1/continuous/train/checkpoint-5") ;;
esac
exec python3 -B -u "$SOURCE/run_main.py" "${ARGS[@]}"
