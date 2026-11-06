#!/bin/bash
set -euo pipefail
ulimit -c 0
unset LD_PRELOAD LD_AUDIT
exec /usr/bin/timeout --kill-after=10s 14980s /usr/local/python3.12/bin/python3 -B -u /public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/GLM-prefill-memory-r472/full-model-entry-v1/controller.py
