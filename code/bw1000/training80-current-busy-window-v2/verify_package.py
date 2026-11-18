"""Exact published sources/ledger; review and adopted states remain explicit."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root,adopted=None):
    root=Path(root);mf=json.loads((root/'SOURCE_FILES.json').read_bytes())
    for name,pin in mf['files'].items():
        if Path(name).name!=name or (root/name).is_symlink():
            raise ValueError('EXACT_FLAT_SOURCE')
        raw=(root/name).read_bytes()
        if len(raw)!=pin['bytes'] or hashlib.sha256(raw).hexdigest()!=pin['SHA256']:
            raise ValueError('PINNED_SOURCE_BYTES')
    c=json.loads((root/'CLAIM.json').read_bytes());pair=(c['root_GO'],c['CPU_GO'],c['GPU_GO'],c['GPU_resource_GO'])
    if pair not in ((False,False,False,False),(True,True,True,True)) or (adopted is not None and pair!=(adopted,adopted,adopted,adopted)):
        raise ValueError('EXPLICIT_CPU_ADOPTION')
    if any(c[k] is not False for k in ('GPU_compute_GO','Model_GO','performance_GO')):
        raise ValueError('NO_GPU_COMPUTE_OR_PROMOTION')
    if c['epoch_seconds']!=180 or c['points']!=3 or c['interval_s']!=2 or c['parents']!=['1087656','1103606']:
        raise ValueError('FROZEN_WINDOW_RESOURCE_SCOPE')
    return mf


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--adopted',action='store_true');a=p.parse_args()
    print(json.dumps(dict(files=len(verify(Path(__file__).resolve().parent,adopted=a.adopted)['files']),verified=True,actual_sampling=False)))
