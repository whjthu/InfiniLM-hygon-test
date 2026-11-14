#!/bin/bash
set -euo pipefail
ulimit -c 0
src="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root=/public/home/aclsylqidf/hygon-inference-refactor-20261014
old=/public/home/aclsylqidf/hygon-inference-20261013
scope="$1"
scratch="$(mktemp -d "/dev/shm/glm-selected-$SLURM_JOB_ID.$SLURM_STEP_ID-$SLURM_PROCID-XXXXXXXX")"
trap 'rm -f -- "$scratch/transport-verbs/libibverbs.so"; rmdir -- "$scratch/transport-verbs" "$scratch" 2>/dev/null || true' EXIT
export HSA_FORCE_FINE_GRAIN_PCIE=1
export APPTAINER_TMPDIR="$scope/tmp-$SLURM_PROCID"
mkdir -m700 "$APPTAINER_TMPDIR"
for name in GLM_SELECTED_CLAIM GLM_SELECTED_EPOCH SLURM_JOB_ID SLURM_STEP_ID SLURM_PROCID; do
    export "APPTAINERENV_$name=${!name}"
done
/public/software/apps/apptainer/1.3.4/bin/apptainer exec --rocm --home "$old/build-deps/root:/root" \
    --bind "$old:/old:ro" --bind "$root:/refactor:ro" --bind "$scope:/refactor/results/${scope##*/}:rw" \
    --bind "$src:/paired-worker:ro" --bind "$scratch:/local" --bind "$APPTAINER_TMPDIR:/tmp" \
    --bind /opt/hyhal:/opt/hyhal:ro --bind "$root/build-deps/rdma-host-stack:/rdma-provider-stack:ro" \
    --bind "$root/candidates/public-shca-layers-r2228/layer3-selected/opt:/candidate-opt:ro" \
    "$root/containers/dtk26-sandbox-r1" /bin/bash --noprofile --norc /paired-worker/inside.sh "/refactor/results/${scope##*/}"
