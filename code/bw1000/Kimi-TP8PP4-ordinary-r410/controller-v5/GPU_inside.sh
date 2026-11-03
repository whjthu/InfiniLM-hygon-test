#!/bin/bash
set -euo pipefail
ulimit -c 0
test "${KIMI410_ORDINARY_RUNTIME:-0}" = 1
# Original epoch is never restarted at container/worker entry.
remaining=$(python3 -c 'import math,os,time;left=math.floor(float(os.environ["KIMI410_SHELL_DEADLINE_UNIX"])-time.time());assert 0<left<=5280;print(left)')
exec timeout --kill-after=20s "${remaining}s" /bin/bash --noprofile --norc /refactor/incoming/Kimi-TP8PP4-ordinary-r410/runtime-v3/target_inside.sh
