"""Distinct namespace/epoch for one previously zero-body official image."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

D=Path(__file__).resolve().parent
G=D.parents[2]
RUN=G/'local-only/training-sf3d-BoomBox-ORM-onefile-completion-v1-original'
REQUEST='e169e9032f7b4f03843117183351bc5b'


def pin(p):
    b=p.read_bytes();return {'bytes':len(b),'SHA256':hashlib.sha256(b).hexdigest()}


def main():
    epoch=time.time();mono=time.monotonic();RUN.mkdir()
    intent={'request_id':REQUEST,'epoch':epoch,'deadline':epoch+90,'max_GETs':1,'max_body_bytes':4471451,
        'original_scope_terminal_preserved':True,'new_namespace_only':True,'source_pins':{p.name:pin(p) for p in D.iterdir() if p.is_file()}}
    (RUN/'intent.json').write_text(json.dumps(intent,sort_keys=True,indent=2)+'\n',encoding='utf8')
    helper=G/'code/bw1000/training-USGS-multievent-render144-v2/job_guard.py'
    if pin(helper)['SHA256']!='c52244b5c239852c09ec2b7c5bd6aa1ffa4056eaf4e822946aaac46c9cd12b71':raise ValueError('qualified Job helper differs')
    s=importlib.util.spec_from_file_location('qualified_single_orm_job',helper);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    job=m.guarded_run([sys.executable,'-B','-X','utf8',str(D/'complete.py'),'--root',str(RUN),'--deadline',str(epoch+90)],RUN/'job-gate.json',RUN,epoch+90,mono+90)
    actual=json.loads((RUN/'result.json').read_bytes()) if (RUN/'result.json').exists() else None
    outcome={'state':'KNOWN_TERMINAL','request_id':REQUEST,'epoch':epoch,'deadline':epoch+90,'Job':job,
        'passed':job['returncode']==0 and actual is not None and actual.get('passed') is True,
        'result':pin(RUN/'result.json') if actual is not None else None,'READY':False}
    (RUN/'outcome.json').write_text(json.dumps(outcome,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print(json.dumps(outcome,sort_keys=True));return 0 if outcome['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
