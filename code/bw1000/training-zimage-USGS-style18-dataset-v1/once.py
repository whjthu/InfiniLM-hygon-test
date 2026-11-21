"""One exclusive local preparation, using the already qualified Windows Job."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

D=Path(__file__).resolve().parent
G=D.parents[2]
RUN=G/'local-only/training-zimage-USGS-style18-dataset-v1-original'
REQUEST='181839cf0cb74a93b96225a94c9c8273'


def main():
    epoch=time.time(); end=time.monotonic()+60
    RUN.mkdir(exist_ok=False)
    sources={p.name:dict(bytes=p.stat().st_size,SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(D.iterdir()) if p.is_file()}
    (RUN/'intent.json').write_text(json.dumps(dict(request_id=REQUEST,epoch=epoch,deadline=epoch+60,
        source=sources,HTTP=0,GPU=0,optimizer_updates=0,target_READY=False),sort_keys=True,indent=2)+'\n',encoding='utf-8')
    for p in D.glob('*.py'):ast.parse(p.read_bytes())
    helper=G/'code/bw1000/training-USGS-multievent-render144-v2/job_guard.py'
    raw=helper.read_bytes()
    assert len(raw)==7659 and hashlib.sha256(raw).hexdigest()=='c52244b5c239852c09ec2b7c5bd6aa1ffa4056eaf4e822946aaac46c9cd12b71'
    spec=importlib.util.spec_from_file_location('qualified_local_job',helper)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    overlay=G/'local-only/workspace-matplotlib-overlay-completion-fsid-fix-v2/overlay'
    argv=[str(D/'prepare.py'),'--output',str(RUN/'dataset'),'--gate',str(RUN/'job-gate.json'),
          '--deadline',str(epoch+60),'--target-data',
          '/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-zimage-USGS-style18-dataset-v1/dataset']
    bootstrap='import sys,runpy;sys.path.insert(0,'+repr(str(overlay))+');sys.argv='+repr(argv)+';runpy.run_path(sys.argv[0],run_name="__main__")'
    job=h.guarded_run([sys.executable,'-B','-X','utf8','-c',bootstrap],RUN/'job-gate.json',RUN,epoch+60,end)
    result_path=RUN/'dataset/result.json'
    r=json.loads(result_path.read_bytes()) if result_path.is_file() else None
    outcome=dict(request_id=REQUEST,epoch=epoch,deadline=epoch+60,Job=job,
        passed=job['returncode']==0 and r is not None and r['passed'] is True,
        result=None if r is None else dict(bytes=result_path.stat().st_size,SHA256=hashlib.sha256(result_path.read_bytes()).hexdigest()),
        actual_HTTP=0,actual_GPU=0,actual_optimizer_updates=0,target_READY=False)
    (RUN/'outcome.json').write_text(json.dumps(outcome,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=outcome['passed'],request_id=REQUEST,Job_rc=job['returncode'],
        completed=0 if r is None else r['completed_images'],planned=18,target_READY=False),sort_keys=True))
    return 0 if outcome['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
