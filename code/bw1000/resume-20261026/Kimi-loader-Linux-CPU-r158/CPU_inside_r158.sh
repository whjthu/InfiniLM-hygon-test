#!/usr/bin/env bash
set -euo pipefail
test "${SLURM_JOB_ID:-}" = 910407
test "${SLURM_CPUS_PER_TASK:-}" = 8
case "$(hostname -s)" in a14r2n20) ;; *) echo 'r158 wrong reserved CPU node' >&2; exit 2;; esac
test "$#" = 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
here="$(cd -- "$(dirname -- "$0")" && pwd)"
exec timeout 175s python3 -B "$here/CPU_entry_r158.py" --out "$1"
