#!/usr/bin/env bash
set -euo pipefail
test "${SLURM_JOB_ID:-}" = 910407
test "${SLURM_CPUS_PER_TASK:-}" = 4
case "$(hostname -s)" in a14r2n20) ;; *) echo 'r159a wrong CPU owner node' >&2; exit 2;; esac
test "$#" = 2
export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
here="$(cd -- "$(dirname -- "$0")" && pwd)"
test "$(sha256sum "$here/source-manifest-r159a.json" | cut -d ' ' -f1)" = "$2"
exec timeout 175s python3 -B "$here/CPU_build_r159a.py" --source-manifest-sha "$2" --out "$1"
