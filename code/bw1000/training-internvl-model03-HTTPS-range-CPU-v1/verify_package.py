"""Static source/AST/claim invariants; never opens a URL, prefix or old receipt."""
import ast,hashlib,json
from pathlib import Path

def verify(directory=None,prepared=False):
    d=Path(directory or Path(__file__).resolve().parent);mf=json.loads((d/'SOURCE_FILES.json').read_bytes())['files']
    assert {p.name for p in d.iterdir() if p.is_file()}==set(mf)|{'SOURCE_FILES.json'}
    for n,pin in mf.items():
        p=d/n;b=p.read_bytes();assert not p.is_symlink() and len(b)==pin['bytes'] and hashlib.sha256(b).hexdigest()==pin['SHA256']
        if p.suffix=='.py':ast.parse(b)
    p=json.loads((d/'PLAN.json').read_bytes());c=json.loads((d/'CLAIM.json').read_bytes())
    assert all(type(c[k])is bool and c[k] is (not prepared) for k in ['root_GO','CPU_GO','HTTP_GO','READ_GO'])
    assert all(c[k] is False for k in ['GPU_GO','Model_GO','training_GO','whole_download_GO'])
    assert p['whole_seconds']==120 and p['worker_stop_seconds']==80 and p['Slurm_max_seconds']==90 and p['Slurm_requested_seconds']==60 and p['independent_read_stop_seconds']==115
    assert p['parent']=='1087656' and p['node']=='a14r3n04' and p['allocated_CPU']==4 and p['worker_CPU']==2 and p['memory_MiB']==2048 and p['GRES']==0
    assert p['commit']=='819e391ba0a9eea8d9d8b1c27cd1e777b0569415' and p['filename']=='model-00003-of-00004.safetensors'
    assert p['URL']=='https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF/resolve/'+p['commit']+'/'+p['filename']
    assert p['range_header']=='bytes=0-1048575' and p['range_bytes']==p['require_Content_Length']==1048576
    assert p['require_Content_Range']=='bytes 0-1048575/4915962496' and p['full_expected_bytes']==4915962496
    assert p['full_expected_SHA256']=='595f3b9043d046ae612f2dbf7cbfcd2465a1c001fcb58e93f78dcaf7cc5976e1'
    assert p['maximum_redirects']==3 and p['maximum_HTTP_attempts']==4 and c['max_invocations']==1 and p['whole_SHA_verified'] is False
    assert hashlib.sha256((d/'held_child.py').read_bytes()).hexdigest()=='06ef9978e7c0957abb2fd0fc6204e16e88a433a966475d53a727bf797ee85351'
    return dict(passed=True,prepared=prepared,files=len(mf),AST_files=len(list(d.glob('*.py'))),probe_HTTP_attempts=0,remote_calls=0,body_reads=0,fixture_executions=0,MF_SHA256=hashlib.sha256((d/'SOURCE_FILES.json').read_bytes()).hexdigest())

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepared',action='store_true');a=p.parse_args();print(json.dumps(verify(prepared=a.prepared)))
