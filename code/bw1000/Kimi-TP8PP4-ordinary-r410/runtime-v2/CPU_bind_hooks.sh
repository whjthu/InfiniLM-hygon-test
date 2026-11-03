# This hook exists only in the GPU-hidden CPU qualification subprocess.
exec() {
  test "$#" = 4
  test "$1" = python
  test "$2" = -B
  test "$3" = -u
  test "$4" = /refactor/incoming/Kimi-TP8PP4-ordinary-r410/runtime-v2/bootstrap.py
  command python -B /refactor/incoming/Kimi-TP8PP4-ordinary-r410/runtime-v2/CPU_provider_bind.py
}
