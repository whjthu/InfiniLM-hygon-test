"""Source-only exact manifest checker; prepared mode does not authorize launch."""
import ast,hashlib,json
from pathlib import Path

def verify(directory=None,prepared=False):
    d=Path(directory or Path(__file__).resolve().parent)
    mf=json.loads((d/'SOURCE_FILES.json').read_bytes())['files']
    assert set(p.name for p in d.iterdir() if p.is_file())==set(mf)|{'SOURCE_FILES.json'}
    for name,pin in mf.items():
        assert Path(name).name==name and not (d/name).is_symlink()
        b=(d/name).read_bytes();assert len(b)==pin['bytes'] and hashlib.sha256(b).hexdigest()==pin['SHA256']
        if name.endswith('.py'):ast.parse(b,filename=name)
    p=json.loads((d/'PLAN.json').read_bytes());c=json.loads((d/'CLAIM.json').read_bytes())
    assert all(type(c[k]) is bool and c[k] is (not prepared) for k in ('root_GO','CPU_GO','READ_GO'))
    assert all(c[k] is False for k in ('GPU_GO','Model_GO','training_GO','provision_GO'))
    assert p['parent']=='1087656' and p['node']=='a14r3n04' and p['allocated_CPU']==4 and p['worker_CPU']==2
    assert p['memory_MiB']==4096 and p['GRES']==0 and p['whole_seconds']==300 and p['Slurm_max_seconds']==240
    rows={r['name']:r for r in p['files']};assert len(rows)==26 and len(p['hash_names'])==len(set(p['hash_names']))==14
    assert sum(rows[n]['bytes'] for n in p['hash_names'])==p['hash_bytes']==4922323586
    assert set(p['old_saved_stat_reference'])==set(p['hash_names']) and c['max_invocations']==1
    assert hashlib.sha256((d/'held_child.py').read_bytes()).hexdigest()=='06ef9978e7c0957abb2fd0fc6204e16e88a433a966475d53a727bf797ee85351'
    return dict(passed=True,source_members=len(mf),prepared=prepared,planned_stat_files=26,planned_hash_files=14,asset_reads=0)
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--prepared',action='store_true');v=a.parse_args();print(json.dumps(verify(prepared=v.prepared),sort_keys=True))
