#!/bin/bash
set -euo pipefail
source /refactor/incoming/GLM-C1-layer-observer-r474/CPU-v1/ENV.sh
exec /usr/bin/timeout --kill-after=5s 570s /usr/bin/python3 -B -u /refactor/incoming/GLM-C1-layer-observer-r474/CPU-v1/build_native.py
