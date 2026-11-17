"""Future per-command WDI environment; original execute_once owns all data authorization."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from overlay_env import G,RUN,OVERLAY,environment

PYTHON=Path('C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
SCRIPT=G/'code/bw1000/training-worldbank-macro-chart-v1/execute_once.py'


def main(args):
    result=json.loads((RUN/'outcome.json').read_bytes())
    if not result['passed'] or Path(sys.executable).resolve()!=PYTHON.resolve():raise ValueError('qualified overlay/fixed Python required')
    if not OVERLAY.is_dir():raise ValueError('original overlay missing')
    # execute_once and its job_guard use sys.executable without-I and default env inheritance.
    # No original WDI source/claim/GO or deadline is changed here.
    return subprocess.run([str(PYTHON),'-B','-X','utf8',str(SCRIPT),*args],env=environment()).returncode


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args().arguments
    raise SystemExit(main(args[1:] if args and args[0]=='--' else args))
