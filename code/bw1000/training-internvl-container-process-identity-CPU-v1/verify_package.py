import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent
def verify(root=D,prepared=False):
    mf=json.loads((root/'SOURCE_FILES.json').read_bytes())['files']
    assert set(mf)=={p.name for p in root.iterdir() if p.is_file() and p.name!='SOURCE_FILES.json'}
    for n,q in mf.items():
        p=root/n;b=p.read_bytes();assert p.name==n and not p.is_symlink() and len(b)==q['bytes'] and hashlib.sha256(b).hexdigest()==q['SHA256']
        if p.suffix=='.py':ast.parse(b,feature_version=(3,10))
    c=json.loads((root/'CLAIM.json').read_bytes());p=json.loads((root/'PLAN.json').read_bytes())
    expected=not prepared
    assert type(prepared) is bool and c['root_GO'] is c['CPU_GO'] is c['READ_GO'] is expected
    assert c['GPU_GO'] is c['Model_GO'] is c['training_GO'] is False and c['max_invocations']==1
    assert p['CPU']==2 and p['memory_MiB']==2048 and p['GRES']==0 and p['whole_seconds']==120 and p['Slurm_seconds']==60 and p['container_max_seconds']==20
    assert mf['held_child.py']['SHA256']=='06ef9978e7c0957abb2fd0fc6204e16e88a433a966475d53a727bf797ee85351'
    assert sum(v['bytes'] for v in mf.values())<262144
    return dict(source_members=len(mf),source_bytes=sum(v['bytes'] for v in mf.values()),prepared=prepared,fixtures=0,target_calls=0)
if __name__=='__main__':
    import argparse
    q=argparse.ArgumentParser();q.add_argument('--prepared',action='store_true');a=q.parse_args();print(json.dumps(verify(prepared=a.prepared)))
