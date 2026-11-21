import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

D=Path(__file__).resolve().parent
G=D.parents[2]
RUN=G/'local-only/training-trellis-public-mesh-quality-input-v1-original'
REQUEST='81416b29dcc443d8aa0dfaf0952d2bba'


def pin(p):
    b=p.read_bytes();return {'bytes':len(b),'SHA256':hashlib.sha256(b).hexdigest()}


def main():
    epoch=time.time();mono=time.monotonic();RUN.mkdir()
    intent={'request_id':REQUEST,'epoch':epoch,'deadline':epoch+90,'max_GETs':3,'exact_total_source_bytes':1019412,'CPU':2,'Job_commit_cap':2147483648,
        'source':{p.name:pin(p) for p in D.iterdir() if p.is_file()},'no_retry_redirect_or_object_reselection':True}
    (RUN/'intent.json').write_text(json.dumps(intent,sort_keys=True,indent=2)+'\n',encoding='utf8')
    guard=G/'code/bw1000/training-USGS-multievent-render144-v2/job_guard.py'
    if pin(guard)['SHA256']!='c52244b5c239852c09ec2b7c5bd6aa1ffa4056eaf4e822946aaac46c9cd12b71':raise ValueError('Job helper differs')
    spec=importlib.util.spec_from_file_location('real_geometry_job',guard);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    job=m.guarded_run([sys.executable,'-B','-X','utf8',str(D/'prepare.py'),'--root',str(RUN),'--deadline',str(epoch+90)],RUN/'job-gate.json',RUN,epoch+90,mono+90)
    r=json.loads((RUN/'result.json').read_bytes()) if (RUN/'result.json').exists() else None
    outcome={'state':'KNOWN_TERMINAL','request_id':REQUEST,'epoch':epoch,'deadline':epoch+90,'Job':job,'passed':job['returncode']==0 and r is not None and r.get('passed') is True,
        'result':pin(RUN/'result.json') if r is not None else None,'READY':False}
    (RUN/'outcome.json').write_text(json.dumps(outcome,sort_keys=True,indent=2)+'\n',encoding='utf8');print(json.dumps(outcome,sort_keys=True))
    return 0 if outcome['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
