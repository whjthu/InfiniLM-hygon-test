"""Offline source verification only; no target probe or implicit adoption."""
import argparse
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parent

def pin(path):
    raw=path.read_bytes();return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())

def verify(adopted):
    for directory in (D,D/'core'):
        mf=json.loads((directory/'SOURCE_FILES.json').read_bytes())
        for name,value in mf['files'].items():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or pin(directory/p)!=value:raise ValueError('member differs')
        claim=json.loads((directory/'CLAIM.json').read_bytes())
        if any(claim[k] is not adopted for k in ('root_GO','GPU_GO','READ_GO','SCORE_GO')):raise ValueError('operative adoption mismatch')
        if any(claim[k] is not False for k in ('training_GO','Model_GO','formal17_GO','READY')):raise ValueError('qualification drift')
    reuse=json.loads((D/'ORIGINAL_CORE_REUSE.json').read_bytes())
    for name,expected in reuse['functional_members'].items():
        if pin(D/'core'/name)!=expected:raise ValueError('frozen original core changed')
    return dict(passed=True,adopted=adopted,outer=pin(D/'SOURCE_FILES.json'),core=pin(D/'core/SOURCE_FILES.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--adopted',action='store_true');a=p.parse_args();print(json.dumps(verify(a.adopted),sort_keys=True))
