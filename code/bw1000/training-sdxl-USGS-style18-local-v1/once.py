"""One exclusive, local-only bounded preparation; no target or Git operations."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

D = Path(__file__).resolve().parent
G = D.parents[2]
REQUEST = '6eef89b3d3ce4842beb9c4fa31f08ed59'
RUN = G / 'local-only/training-sdxl-USGS-style18-local-v1-original'


def main():
    origin, mono = time.time(), time.monotonic()
    RUN.mkdir(exist_ok=False)
    source = {p.name: dict(bytes=len(p.read_bytes()), SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in sorted(D.iterdir()) if p.is_file()}
    for p in D.glob('*.py'):
        ast.parse(p.read_bytes(), filename=str(p))
    (RUN / 'intent.json').write_text(json.dumps(dict(request_id=REQUEST, epoch=origin, deadline=origin+60,
        source=source, GPU=0, HTTP=0, target_READY=False), sort_keys=True, indent=2)+'\n', encoding='utf-8')
    helper = G / 'code/bw1000/training-USGS-multievent-render144-v2/job_guard.py'
    expected = json.loads((D / 'LOCAL_SCOPE.json').read_bytes())['qualified_job_helper']
    raw = helper.read_bytes()
    if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['SHA256']:
        raise ValueError('qualified Job helper changed')
    spec = importlib.util.spec_from_file_location('style18_job', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    overlay = G / 'local-only/workspace-matplotlib-overlay-completion-fsid-fix-v2/overlay'
    args = [str(D/'prepare.py'), '--output', str(RUN/'dataset'), '--gate', str(RUN/'job-gate.json'), '--deadline', str(origin+60)]
    bootstrap = 'import sys,runpy;sys.path.insert(0,'+repr(str(overlay))+');sys.argv='+repr(args)+';runpy.run_path(sys.argv[0],run_name="__main__")'
    job = module.guarded_run([sys.executable, '-B', '-X', 'utf8', '-c', bootstrap], RUN/'job-gate.json', RUN, origin+60, mono+60)
    result_path = RUN / 'dataset/result.json'
    result = json.loads(result_path.read_bytes()) if result_path.is_file() else None
    outcome = dict(request_id=REQUEST, epoch=origin, deadline=origin+60, Job=job,
                   passed=job['returncode']==0 and result is not None and result['passed'] is True,
                   result_file=None if result is None else dict(bytes=result_path.stat().st_size, SHA256=hashlib.sha256(result_path.read_bytes()).hexdigest()),
                   target_READY=False, GPU_GO=False, training_GO=False, HTTP=0, GPU=0)
    (RUN/'outcome.json').write_text(json.dumps(outcome, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=outcome['passed'], request_id=REQUEST, job_rc=job['returncode'],
                         completed=0 if result is None else result['completed_images'], planned=18, target_READY=False)))
    return 0 if outcome['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
