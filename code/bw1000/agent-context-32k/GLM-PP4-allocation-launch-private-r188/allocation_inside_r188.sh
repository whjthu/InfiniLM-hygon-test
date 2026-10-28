#!/bin/bash
set -euo pipefail
python3 -B /refactor/incoming/GLM-PP4-allocation-launch-r188/publish_step_GO_r188.py
exec /bin/bash /refactor/incoming/GLM-PP4-capacity-r186/GPU_inside_r186.sh
