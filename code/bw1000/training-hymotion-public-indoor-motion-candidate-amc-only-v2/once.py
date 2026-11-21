"""One bounded local public-data acquisition. No cluster/model calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

D=Path(__file__).resolve().parent
G=D.parents[2]
RUN=G/'local-only/training-hymotion-public-indoor-motion-candidate-amc-only-v2-original'
REQUEST='85439889fa1d4f398a4adbe69531a088'
HELPER=G/'code/bw1000/training-USGS-multievent-render144-v2/job_guard.py'
HELPER_SHA='c52244b5c239852c09ec2b7c5bd6aa1ffa4056eaf4e822946aaac46c9cd12b71'


def pin(path):
    b=path.read_bytes();return dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())


def main():
    epoch=time.time();mono=time.monotonic();deadline=epoch+180
    RUN.mkdir()
    intent=dict(request_id=REQUEST,epoch=epoch,deadline=deadline,planned_requests_max=1,source_bytes_cap=5*1024**2,
        source_files={p.name:pin(p) for p in sorted(D.iterdir()) if p.is_file()},no_retry=True,no_redirect=True,
        TLS_source_authentication=False,root_GPU_or_cluster_calls=0)
    (RUN/'intent.json').write_text(json.dumps(intent,sort_keys=True,indent=2)+'\n',encoding='utf8')
    outcome=dict(request_id=REQUEST,epoch=epoch,deadline=deadline,passed=False,actual_training_or_GPU=0,READY=False)
    try:
        if pin(HELPER)['SHA256']!=HELPER_SHA:raise ValueError('Qualified Job helper source differs')
        spec=importlib.util.spec_from_file_location('qualified_local_job',HELPER)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        job=module.guarded_run([sys.executable,'-B','-X','utf8',str(D/'acquire_parse.py'),'--root',str(RUN),
            '--deadline',str(deadline),'--gate',str(RUN/'job-gate.json')],RUN/'job-gate.json',RUN,deadline,mono+180)
        outcome['Job']=job
        if (RUN/'result.json').exists():
            result=json.loads((RUN/'result.json').read_bytes());outcome['worker_result']=pin(RUN/'result.json')
            outcome['passed']=job['returncode']==0 and result.get('passed') is True
        outcome['state']='KNOWN_TERMINAL'
    except Exception as e:outcome.update(state='LOCAL_BOUNDARY_FAILURE',error_type=type(e).__name__,reason=str(e)[:512])
    outcome['wall_seconds']=time.monotonic()-mono
    (RUN/'outcome.json').write_text(json.dumps(outcome,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print(json.dumps(outcome,sort_keys=True))
    return 0 if outcome['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
