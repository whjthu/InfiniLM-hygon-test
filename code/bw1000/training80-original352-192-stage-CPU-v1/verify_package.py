import argparse
import hashlib
import json
from pathlib import Path


def verify(root,adopted=None):
    root=Path(root);mf=json.loads((root/'SOURCE_FILES.json').read_bytes())
    for name,pin in mf['files'].items():
        if Path(name).name!=name or (root/name).is_symlink():raise ValueError('FLAT_EXACT_SOURCE')
        raw=(root/name).read_bytes()
        if len(raw)!=pin['bytes'] or hashlib.sha256(raw).hexdigest()!=pin['SHA256']:raise ValueError('SOURCE_SHA')
    c=json.loads((root/'CLAIM.json').read_bytes());pair=(c['root_GO'],c['CPU_GO'])
    if pair not in ((False,False),(True,True)) or (adopted is not None and pair!=(adopted,adopted)):raise ValueError('EXPLICIT_CPU_ADOPTION')
    if c['GPU_GO'] is not False or c['sampler_GO'] is not False or c['epoch_seconds']!=90:raise ValueError('NO_SAMPLING_OR_GPU')
    if c['node']!='a01r4n14' or c['parent']!='1087656' or c['busy_counter_reads']!=0:raise ValueError('EXACT_SINGLE_NODE_METADATA')
    return mf


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--adopted',action='store_true');a=p.parse_args()
    print(json.dumps(dict(verified=True,files=len(verify(Path(__file__).resolve().parent,adopted=a.adopted)['files']),actual_remote=False)))
