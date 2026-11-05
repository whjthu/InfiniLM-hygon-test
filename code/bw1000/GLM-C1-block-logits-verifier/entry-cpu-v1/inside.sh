#!/bin/bash
set -euo pipefail
ulimit -c 0
root=/refactor/incoming/GLM952-prefix-continuation
original=/refactor/incoming/GLM-W4-remapped-final-slots-r469/deployment-v1/inside.sh
test "$(sha256sum "$original" | cut -d' ' -f1)" = 418dc5466f3f91c14c8eecba753f45340bf85d7ba4d4f0c57f8c2a8df66d701b
export GLM_R444_ENV_ONLY=1 GLM_EXACT_BANKS_R4188=1 GLM_CHANNELS_R4363=0
export GLM_BOUNDED_ROPE_R4279=1 GLM_SHARED_CONTROL_R4279=1 GLM_NETWORK_TRIAL=shca
export INFINILM_OWNER_CPU_WIDTH_R2392=4
exec /usr/bin/timeout --kill-after=5s 105s /usr/bin/python3 -B "$root/entry-cpu-v1/environment.py" "$root"
