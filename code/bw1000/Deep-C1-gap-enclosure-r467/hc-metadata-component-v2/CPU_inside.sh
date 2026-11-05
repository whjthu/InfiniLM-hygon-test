#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
unset LD_PRELOAD BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch.libs:${LD_LIBRARY_PATH:-}
exec timeout --kill-after=10s 270 python3 -B /refactor/incoming/Deep-C1-gap-enclosure-r467/hc-metadata-component-v2/build_CPU.py
