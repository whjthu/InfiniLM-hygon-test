#!/bin/bash
set -euo pipefail
ulimit -c 0
unset LD_PRELOAD LD_AUDIT
exec /usr/bin/timeout --kill-after=5s 295s /usr/bin/python3 -B -u /public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/GLM952-global-TP16-component-r470/device-v2/controller.py
