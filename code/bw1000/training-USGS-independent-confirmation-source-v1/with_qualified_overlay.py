"""Run only this offline USGS once under the existing exact qualified child environment."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

G=Path(__file__).resolve().parents[3]
PYTHON=Path('C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
ENV=G/'code/bw1000/workspace-matplotlib-overlay-completion-v1/overlay_env.py'
OUTCOME=G/'local-only/workspace-matplotlib-overlay-completion-fsid-fix-v2/outcome.json'


def main(args):
    epoch=time.time()
    if OUTCOME.is_symlink() or OUTCOME.stat().st_size!=OUTCOME_BYTES:raise ValueError('Original renderer outcome size/type changed')
    raw=OUTCOME.read_bytes();result=json.loads(raw)
    if len(raw)!=OUTCOME_BYTES or hashlib.sha256(raw).hexdigest()!=OUTCOME_SHA:raise ValueError('Original qualified renderer outcome changed')
    if result['passed'] is not True or result['job']['returncode']!=0 or result['job']['timed_out'] is not False:raise ValueError('Renderer not actually qualified')
    if Path(sys.executable).resolve()!=PYTHON.resolve():raise ValueError('Fixed Python required')
    if ENV.is_symlink() or ENV.stat().st_size!=712:raise ValueError('Environment provider size/type changed')
    source=ENV.read_bytes()
    if len(source)!=712 or hashlib.sha256(source).hexdigest()!=ENV_SHA:raise ValueError('Published environment provider changed')
    # Import only the exact already-published stdlib environment provider; it does not import renderer/model.
    spec=importlib.util.spec_from_file_location('usgs_chart_qualified_overlay_env',ENV);module=importlib.util.module_from_spec(spec)
    exec(compile(source,str(ENV),'exec'),module.__dict__)
    if not module.OVERLAY.is_dir():raise ValueError('Actual qualified overlay missing')
    return subprocess.run([str(PYTHON),'-B','-X','utf8',str(Path(__file__).with_name('once.py')),*args,'--origin',str(epoch)],env=module.environment(),timeout=max(.01,epoch+120-time.time())).returncode


# Filled from exact already-known local qualification bytes, not a new install/GET.
ENV_SHA='0daf99cb192c9058381c4c9e1916daa65d092fdf7ac664a95efa420d7321a646'
OUTCOME_SHA='d4642ac075a3681e2b13611d5c43858f591f36919ccd921afbf431447f6b0c2a'
OUTCOME_BYTES=13615

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args().arguments
    raise SystemExit(main(a[1:] if a and a[0]=='--' else a))
