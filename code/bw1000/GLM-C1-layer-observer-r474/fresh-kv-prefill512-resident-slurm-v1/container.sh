#!/bin/bash
set -euo pipefail
ulimit -c 0
if [[ ${1:-} == --inside ]]; then
    shift
    source /stage/ENV.sh
    unset GLM_C1_LAYER_OBSERVER_R474 GLM_HISTORY_VALUE_ONLY_V1 GLM_SELECTED_VALUE_PACKED_LAYER0 INFINILM_GLM474_FRESH_KV_PACK
    export GLM_GLOBAL_TP16_R470=1 GLM_GLOBAL_TP16_MODEL_R470=1
    export GLM_V_ON_DEMAND_R403=1 GLM_BOUNDED_ROPE_R4279=1 GLM_SHARED_CONTROL_R4279=1 GLM_VOCAB_PARALLEL_EMBEDDING_R471=1
    export MKL_NUM_THREADS=1
    exec /usr/bin/python3 -B -u /stage/worker.py node "/refactor/results/${1##*/}"
fi
root=/public/home/aclsylqidf/hygon-inference-refactor-20261014
old=/public/home/aclsylqidf/hygon-inference-20261013
scope="${1:?expected new experiment scope}"
stage="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scratch="$(mktemp -d "/dev/shm/glm474-t512-${SLURM_JOB_ID}.${SLURM_STEP_ID}-XXXXXXXX")"
trap 'rmdir -- "$scratch"' EXIT
export APPTAINER_TMPDIR="$scratch" HSA_FORCE_FINE_GRAIN_PCIE=1 APPTAINERENV_TMPDIR=/tmp
/public/software/apps/apptainer/1.3.4/bin/apptainer exec --rocm --home "$old/build-deps/root:/root" \
    --bind "$old:/old:ro" --bind "$root:/refactor:ro" --bind "$scope:/refactor/results/${scope##*/}:rw" \
    --bind "$stage:/stage:ro" --bind "$scratch:/local" --bind /opt/hyhal:/opt/hyhal:ro \
    --bind /public/home/aclsylqidf/shared:/shared:ro \
    "$root/containers/dtk26-sandbox-r1" /bin/bash --noprofile --norc /stage/container.sh --inside "$scope"
