#!/usr/bin/env bash
set -euo pipefail
source /opt/dtk/env.sh
cd /refactor/incoming/Kimi-W13-wavepartial-component-r155
: "${MANIFEST_R155:?root-published exact GPU manifest required}"
exec timeout 600s python3 -B -u run_GPU_r155.py "$MANIFEST_R155"
