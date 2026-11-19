"""Stdlib package/AST invariants only. Never reads private reference, weights or receipts."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent

def verify(root=D):
    pins=json.loads((root/'SOURCE_FILES.json').read_bytes())['files']
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert actual==set(pins)|{'SOURCE_FILES.json'}
    for name,pin in pins.items():
        path=root/name;raw=path.read_bytes();assert not path.is_symlink() and len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
        if path.suffix=='.py':ast.parse(raw)
    p=json.loads((root/'PLAN.json').read_bytes());c=json.loads((root/'CLAIM.json').read_bytes())
    assert p['case']==c['case']=='T03.sft' and c['max_invocations']==1 and p['whole_seconds']==c['whole_seconds']==900
    assert p['root_GO'] is c['root_GO'] and p['provision_GO'] is c['provision_GO'] and type(c['root_GO'])is bool and type(c['provision_GO'])is bool
    assert not c['GPU_GO'] and not c['Model_GO'] and not c['training_GO']
    assert [x['name'] for x in p['files']]==['reference/input-fingerprints.json','initial/rank-00/initial.pt']
    assert sum(x['bytes'] for x in p['files'])==p['stream_payload_bytes']==c['payload_bytes']==174847256<192*1024**2
    assert p['files'][1]['bytes']==174821899 and p['files'][1]['SHA256']=='2a0df4d5fdd4d793d47859f7e6ee9102cd3324199f90d0ed34760d3c7198612f'
    assert p['local_reference']['bytes']==25357 and p['local_reference']['SHA256']=='a52d848f052dfc0ccb42335a7ffa6a1982723fc2a82064fc612d7e1bfca1d959'
    assert p['target_parent']=='1087656' and p['target_node']=='a14r3n04' and p['source_CPU']==2 and p['target_CPU']==4 and p['target_worker_CPU']==2 and p['target_memory_MiB']==4096 and p['target_GRES']==0
    assert hashlib.sha256((root/'held_child.py').read_bytes()).hexdigest()=='06ef9978e7c0957abb2fd0fc6204e16e88a433a966475d53a727bf797ee85351'
    return dict(passed=True,files=len(pins),AST_files=len(list(root.rglob('*.py'))),root_GO=c['root_GO'],provision_GO=c['provision_GO'],READ_GO=c['READ_GO'],actual_remote=0,actual_private_body_reads=0,fixture_executions=0,MF_SHA256=hashlib.sha256((root/'SOURCE_FILES.json').read_bytes()).hexdigest())

if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
