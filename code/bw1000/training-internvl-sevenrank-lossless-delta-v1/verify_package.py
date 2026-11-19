"""Stdlib AST/source-manifest invariants only; no codec fixture or private asset access."""
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
    for key in ['root_GO','provision_GO','READ_GO','GPU_GO','Model_GO','training_GO']:
        assert type(c[key])is bool and p[key] is c[key]
    assert not c['GPU_GO'] and not c['Model_GO'] and not c['training_GO']
    assert [r['rank'] for r in p['source_files']]==list(range(8)) and p['source_files'][1:]==p['files']
    assert [r['name'] for r in p['files']]==['initial/rank-%02d/initial.pt'%rank for rank in range(1,8)]
    assert all(r['bytes']==174821899 and len(r['SHA256'])==64 for r in p['source_files']) and len({r['SHA256'] for r in p['source_files']})==8
    assert all(p['base'][k]==p['source_files'][0][k] for k in p['source_files'][0])
    assert p['base']['SHA256']=='2a0df4d5fdd4d793d47859f7e6ee9102cd3324199f90d0ed34760d3c7198612f'
    assert p['base']['target']=='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-internvl-rank00-reference-provision-v1/payload/initial/rank-00/initial.pt'
    assert p['source_full_read_bytes']==c['source_full_read_bytes']==sum(r['bytes'] for r in p['source_files'])==1398575192
    assert p['target_output_bytes']==c['target_output_bytes']==sum(r['bytes'] for r in p['files'])==1223753293
    assert p['target_base_read_bytes']==c['target_base_read_bytes']==174821899
    assert p['block_bytes']==65536 and p['frame_cap_bytes']==c['frame_cap_bytes']==4194304 and p['wire_cap_bytes']==c['wire_cap_bytes']==29360128
    assert p['target_parent']=='1087656' and p['target_node']=='a14r3n04' and p['source_CPU']==2 and p['target_CPU']==4 and p['target_worker_CPU']==2 and p['target_memory_MiB']==4096 and p['target_GRES']==0
    assert p['raw_fallback'] is c['raw_fallback'] is False and p['replay_allowed'] is c['replay_allowed'] is False
    assert hashlib.sha256((root/'held_child.py').read_bytes()).hexdigest()=='06ef9978e7c0957abb2fd0fc6204e16e88a433a966475d53a727bf797ee85351'
    assert sum((root/name).stat().st_size for name in [*pins,'SOURCE_FILES.json'])<=196608
    return dict(passed=True,files=len(pins),AST_files=len(list(root.rglob('*.py'))),root_GO=c['root_GO'],provision_GO=c['provision_GO'],READ_GO=c['READ_GO'],actual_remote=0,actual_private_body_reads=0,fixture_executions=0,MF_SHA256=hashlib.sha256((root/'SOURCE_FILES.json').read_bytes()).hexdigest())

if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
