#!/bin/bash
set -euo pipefail

area=/source/code/bw1000/GLM-C1-layer-observer-r474
env_file="$area/depth-stratified-observer-CPU-v1/ENV.sh"
printf '%s  %s\n' 147ea951bfe67a82ba39ef62e2e695e4ae7d21f8cc03f95b60054527ed960db9 "$env_file" |
    /usr/bin/sha256sum --check --status

# Preserve Slurm's exact GPU masks through the original CPU-only ENV.
declare -A inherited_masks
for mask in CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do
    if [[ -v $mask ]]; then
        inherited_masks[$mask]=${!mask}
    fi
done
source "$env_file"
for mask in CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do
    if [[ -v inherited_masks[$mask] ]]; then
        export "$mask=${inherited_masks[$mask]}"
    else
        unset "$mask"
    fi
done

seconds=$(/usr/bin/python3 -I -S -B -c '
import json, math, time
with open("/scope/intent.json", "rb") as handle:
    raw = handle.read(16385)
if not 0 < len(raw) <= 16384:
    raise ValueError("invalid intent size")
intent = json.loads(raw)
epoch, work, whole = (intent[key] for key in ("epoch", "work_deadline", "deadline"))
now = time.time()
if not math.isfinite(epoch) or not 0 < epoch <= now or work != epoch+270 or whole != epoch+300:
    raise ValueError("wrong original device epoch")
seconds = math.floor(work-now)-5
if seconds < 1:
    raise SystemExit("no work budget before timeout kill reserve")
print(seconds)
')
exec /usr/bin/timeout --kill-after=5s "${seconds}s" \
    /usr/bin/python3 -I -S -B -u "$area/fresh-kv-pack-device-numeric-v2/probe.py" \
    --intent /scope/intent.json --pre-admission /scope/before.json --out /scope/probe.jsonl
