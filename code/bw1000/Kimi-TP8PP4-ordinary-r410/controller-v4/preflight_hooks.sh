# Only a CPU preflight child receives this BASH_ENV. GPU clean_env removes it.
# Execute the actual frozen production shell; intercept only its final exec leaf.
exec() {
  test "$#" = 4
  test "$1" = python
  test "$2" = -B
  test "$3" = -u
  test "$4" = /refactor/incoming/Kimi-TP8PP4-ordinary-r410/runtime-v3/bootstrap.py
  KIMI410_PREFLIGHT_FINAL_ARGC="$#" KIMI410_PREFLIGHT_FINAL_COMMAND="$*" command python -B "${KIMI410_PREFLIGHT_CONSUMER:?}"
}
