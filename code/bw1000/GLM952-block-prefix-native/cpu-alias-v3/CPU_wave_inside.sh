#!/bin/bash
set -euo pipefail
root=/refactor/incoming/GLM952-block-prefix-native/cpu-alias-v3
exec /bin/bash "$root/CPU_inside_prefix_build.sh" "$root/CPU_RESOURCE_CLAIM.json" "$GLM952_PREFIX_CPU_CLAIM_SHA"
