"""One local public acquisition+decode/target bridge; no default remote route."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

D=Path(__file__).resolve().parent
G=D.parents[2]
RUN=G/'local-only/training-sf3d-public-pbr-atlas2-v1-original'
REQUEST='3425bcd8f8884785b23ef0192e187f39'


def pin(p):
    b=p.read_bytes();return dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())


def main():
    epoch=time.time();mono=time.monotonic();RUN.mkdir()
    intent=dict(request_id=REQUEST,epoch=epoch,deadline=epoch+180,max_GETs=10,max_body_bytes=16777216,CPU=2,commit_cap=2147483648,
        source_pins={p.name:pin(p) for p in D.iterdir() if p.is_file()},train_object='WaterBottle',dev_object='BoomBox',blind_objects=0,
        no_redirect_retry_fallback=True,original_T11_checkpoint_or_8_RGBA_read=False)
    (RUN/'intent.json').write_text(json.dumps(intent,sort_keys=True,indent=2)+'\n',encoding='utf8')
    helper=G/'code/bw1000/training-USGS-multievent-render144-v2/job_guard.py'
    if pin(helper)['SHA256']!='c52244b5c239852c09ec2b7c5bd6aa1ffa4056eaf4e822946aaac46c9cd12b71':raise ValueError('Job helper differs')
    spec=importlib.util.spec_from_file_location('qualified_local_job',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    job=m.guarded_run([sys.executable,'-B','-X','utf8',str(D/'acquire_build.py'),'--root',str(RUN),'--deadline',str(epoch+180)],RUN/'job-gate.json',RUN,epoch+180,mono+180)
    result=json.loads((RUN/'result.json').read_bytes()) if (RUN/'result.json').exists() else None
    outcome=dict(request_id=REQUEST,epoch=epoch,deadline=epoch+180,state='KNOWN_TERMINAL',Job=job,passed=job['returncode']==0 and result is not None and result.get('passed') is True,
        result=pin(RUN/'result.json') if result is not None else None,READY=False)
    (RUN/'outcome.json').write_text(json.dumps(outcome,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print(json.dumps(outcome,sort_keys=True));return 0 if outcome['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
