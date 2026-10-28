#!/bin/bash
set -euo pipefail
python3 -B /refactor/incoming/GLM-PP4-allocation-launch-r182b/publish_step_GO_r182b.py
exec /bin/bash /refactor/incoming/GLM-PP4-capacity-r178/GPU_inside_r178.sh
