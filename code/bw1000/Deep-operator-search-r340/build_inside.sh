#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
cd /refactor/incoming/Deep-operator-search-r340
test ! -e lib340.so
hipcc -O3 -fPIC -shared --offload-arch=gfx936 -ffp-contract=off kernels.hip -o lib340.so
sha256sum kernels.hip lib340.so
ldd lib340.so
hipcc --version
