"""Exact local evidence inventory, never remote status or implicit GPU readiness."""
import argparse
import hashlib
import json
from pathlib import Path

def inspect(repo):
    registry=json.loads((Path(__file__).parent/'CANDIDATES.json').read_bytes());evidence={}
    for key,pin in registry['evidence'].items():
        p=repo/pin['path'];raw=p.read_bytes()
        if len(raw)!=pin['bytes'] or hashlib.sha256(raw).hexdigest()!=pin['SHA256']:raise ValueError('evidence changed: '+key)
        evidence[key]=True
    result=[]
    for r in registry['candidates']:
        if r['training_dispatch_READY'] is not False:raise ValueError('this frozen snapshot has no fully ready new GPU scope; root needs actual admission evidence')
        result.append({k:r[k] for k in ('id','model','case','domain','stage','training_dispatch_READY','next_action','blockers')})
    return dict(candidate_count=5,fully_ready_new_training_scopes=0,evidence_files_verified=len(evidence),candidates=result,
        remote_calls=0,GPU_calls=0,allocation_equals_busy=False,root_dispatch_required=True)
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--repo',type=Path,required=True);print(json.dumps(inspect(q.parse_args().repo),ensure_ascii=False,sort_keys=True))
