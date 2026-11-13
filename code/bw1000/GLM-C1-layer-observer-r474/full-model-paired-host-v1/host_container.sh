#!/bin/bash
set -euo pipefail
ulimit -c 0
src="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root=/public/home/aclsylqidf/hygon-inference-refactor-20261014
old=/public/home/aclsylqidf/hygon-inference-20261013
scope="${1:?missing run directory}"
scratch="$(mktemp -d "/dev/shm/glm474-${SLURM_JOB_ID}.${SLURM_STEP_ID}-${SLURM_PROCID}-XXXXXXXX")"
# These small node-local diagnostics remain available for postmortem on failure.
export HSA_FORCE_FINE_GRAIN_PCIE=1
export APPTAINER_TMPDIR="$scope/tmp-${SLURM_PROCID}"
mkdir -m700 "$APPTAINER_TMPDIR"
export APPTAINERENV_TMPDIR=/tmp
for name in GLM474_PAIR_ID GLM474_PAIR_ARM GLM470_EPOCH GLM470_GPU_MF; do
    export "APPTAINERENV_${name}=${!name}"
done
exec /public/software/apps/apptainer/1.3.4/bin/apptainer exec --rocm \
    --home "$old/build-deps/root:/root" \
    --bind "$old:/old:ro" --bind "$root:/refactor:ro" \
    --bind "$scope:/refactor/results/${scope##*/}:rw" \
    --bind "$src:/paired-worker:ro" --bind "$scratch:/local" \
    --bind "$APPTAINER_TMPDIR:/tmp" --bind /opt/hyhal:/opt/hyhal:ro \
    --bind /public/home/aclsylqidf/shared:/shared:ro \
    "$root/containers/dtk26-sandbox-r1" /bin/bash --noprofile --norc /paired-worker/inside.sh
