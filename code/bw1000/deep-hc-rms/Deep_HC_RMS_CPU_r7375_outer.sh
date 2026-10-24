#!/bin/bash
set -euo pipefail
ulimit -c 0
R=/public/home/aclsylqidf/hygon-inference-refactor-20261014
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES=''
/public/software/apps/apptainer/1.3.4/bin/apptainer exec --home /public/home/aclsylqidf/hygon-inference-20261013/build-deps/root:/root --bind "$R:/refactor" "$R/containers/dtk26-sandbox-r1" bash /refactor/incoming/Deep_HC_RMS_CPU_r7375_inside.sh
