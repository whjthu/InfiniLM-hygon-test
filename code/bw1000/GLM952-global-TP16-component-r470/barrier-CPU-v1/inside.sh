#!/bin/bash
set -euo pipefail
ulimit -c 0
unset LD_PRELOAD LD_AUDIT BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT
export PYTHONDONTWRITEBYTECODE=1
exec /usr/bin/python3 -B -u /refactor/incoming/GLM952-global-TP16-component-r470/barrier-CPU-v1/barrier.py worker
