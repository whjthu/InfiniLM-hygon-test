"""Only new source/AST/MF and frozen metadata arithmetic, never retests old qualification or bodies."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent

def verify(root=D):
    raw=(root/'SOURCE_FILES.json').read_bytes();pins=json.loads(raw)['files'];actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts};assert actual==set(pins)|{'SOURCE_FILES.json'}
    for n,v in pins.items():
        p=root/n;b=p.read_bytes();assert not p.is_symlink() and len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['SHA256']
        if p.suffix=='.py':ast.parse(b,filename=n)
    p=json.loads((root/'PLAN.json').read_bytes());c=json.loads((root/'CLAIM.json').read_bytes());d=json.loads((root/'DEPENDENCIES.json').read_bytes())
    assert p['schema']==c['schema']=='INTERNVL_MODELROOT17_CPU_ASSEMBLY_V1' and p['state']==c['state']=='SOURCE_PREPARED_BLOCKED_BY_FAILED_PREREQUISITE'
    for k in ['root_GO','CPU_GO','assembly_GO','READ_GO','GPU_GO','Model_GO','training_GO']:assert type(p[k])is bool and p[k] is c[k]
    assert not any(c[k] for k in ['GPU_GO','Model_GO','training_GO']) and c['max_invocations']==1 and c['execution_READY'] is c['original_prerequisite_passed'] is False
    assert p['whole_seconds']==c['whole_seconds']==900 and p['chunk_bytes']==c['buffer_bytes_max']==1048576
    assert len(p['read3'])==3 and len(p['copy14'])==14 and len({r['name'] for r in p['read3']+p['copy14']})==17
    assert sum(r['bytes'] for r in p['read3'])==p['read3_bytes']==12150388664
    assert sum(r['bytes'] for r in p['copy14'])==p['copy14_bytes']==4922323586 and p['total_read_bytes']==c['read_bytes']==17072712250 and p['total_write_bytes']==c['write_bytes']==4922323586
    assert all(r['name'].startswith('model/') and '/initial/' not in r['source'] and '.incomplete' not in r['source'] for r in p['read3']+p['copy14'])
    assert p['parent']=='1087656' and p['node']=='a14r3n04' and p['allocated_CPU']==4 and p['worker_CPU']==2 and p['memory_MiB']==4096 and p['GRES']==0
    ctx=d['original_context'];assert ctx['publication']=='0c962c1deb86497f5efbcb80230037783512a3bb' and ctx['request_id']=='328b0f23c38949f1a9149f0006e48b9b' and ctx['epoch']==1795057383.2257993 and ctx['deadline']==1795143783.2257993 and ctx['namespace_nonce']=='549a9cbfa6e1473ab85a2d8bf712666c' and ctx['source_manifest_SHA256']=='a36dbe5c23f495713b118793b92fde3cc7014d424ce7339574896e28cdc6318f'
    assert d['substitution_allowed'] is d['alternate_context_CLI'] is d['source_prepared_is_READY'] is False
    assert d['original_intent']['bytes']==434 and d['original_intent']['SHA256']=='dbc1780fae7765ca714391c759c9708f6dd37ec8055e236c5bdf294da5ef3182'
    assert list(p['budget'].values())==sorted(p['budget'].values()) and p['budget']['whole_end']==900
    assert hashlib.sha256((root/'held_child.py').read_bytes()).hexdigest()=='06ef9978e7c0957abb2fd0fc6204e16e88a433a966475d53a727bf797ee85351'
    assert sum((root/n).stat().st_size for n in [*pins,'SOURCE_FILES.json'])<=196608
    return dict(passed=True,files=len(pins),AST_files=len(list(root.glob('*.py'))),root_GO=c['root_GO'],CPU_GO=c['CPU_GO'],assembly_GO=c['assembly_GO'],READ_GO=c['READ_GO'],MF_SHA256=hashlib.sha256(raw).hexdigest(),execution_READY=False,actual_body_reads=0,actual_remote=0,old_qualification_tests=0,fixture_executions=0)

if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
