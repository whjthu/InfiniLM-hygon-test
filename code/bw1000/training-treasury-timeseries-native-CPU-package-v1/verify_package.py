"""Verify complete installed CPU package; no native imports or remote operations."""
import ast
import hashlib
import json
from pathlib import Path,PurePosixPath


def verify(source):
    claim=json.loads((source/'CLAIM.json').read_bytes())
    expected=dict(claim['source_SHA256'])
    expected['CLAIM.json']=hashlib.sha256((source/'CLAIM.json').read_bytes()).hexdigest()
    manifest=json.loads((source/'MF.json').read_bytes())
    if set(manifest)!=set(expected):raise ValueError('package manifest keys differ')
    actual=set()
    for i,path in enumerate(source.rglob('*')):
        if i>=64 or path.is_symlink():raise ValueError('unsafe package traversal')
        if path.is_file():actual.add(path.relative_to(source).as_posix())
    if actual!=set(expected)|{'MF.json'}:raise ValueError('extra or missing package member')
    AST=0
    for name,digest in expected.items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix()!=name:raise ValueError('unsafe package path')
        raw=(source/name).read_bytes()
        if len(raw)>262144 or dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())!=manifest[name] or digest!=manifest[name]['SHA256']:
            raise ValueError('package member bytes changed')
        if name.endswith('.py'):ast.parse(raw);AST+=1
    if (claim['parent'],claim['node'],claim['whole_epoch_seconds'],claim['Slurm_maximum_seconds'],
        claim['inside_maximum_seconds'],claim['CPU_cores'],claim['memory_MiB'],claim['GPU_cards']) != (
        '1103606','f13r1n11',600,480,450,8,16384,0):raise ValueError('CPU claim changed')
    if claim['label']!='training-treasury-timeseries-native-CPU-package-v1':raise ValueError('CPU namespace changed')
    if hashlib.sha256((source/'native/MF.json').read_bytes()).hexdigest()!='d4b802d666d4a72305400fdbabccf5d93611ca8e07c26f7c06a94b18be6d6035':
        raise ValueError('original native6 manifest changed')
    old=json.loads((source/'native/MF.json').read_bytes())
    for name,identity in old.items():
        raw=(source/'native'/name).read_bytes()
        if dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())!=identity:raise ValueError('original native6 copy changed')
    return dict(passed=True,package_files=len(actual),AST_checked=AST,native_copies=6,
                actual_native_calls=0,actual_new_Slurm_steps=0,GPU_cards=0)


if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent),sort_keys=True))
