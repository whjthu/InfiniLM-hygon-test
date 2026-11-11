"""Independently rehash all installed source and inspect the actual long allocation."""
import json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent;task=json.loads((BASE/'task.json').read_bytes())
manifest=json.loads((BASE/'FRAMEWORK_FILES.json').read_bytes());ns={'__file__':str(Path(__file__).resolve())}
prelude=(BASE.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8')
exec(compile(prelude.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
code=r'''
import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-framework-8a487b2-v1')
raw=(root/'receipt.json').read_bytes();receipt=json.loads(raw)
assert receipt['claim_commit']=='a07ecaca5135d94de8d3456362770970e19bf5bb' and receipt['entry_returncode']==0
assert receipt['deadline']-receipt['epoch']==90 and receipt['framework_commit']==MANIFEST['framework_commit'] and receipt['framework_tree']==MANIFEST['framework_tree']
assert receipt['files_verified']==len(MANIFEST['files'])==271
mf=(root/'FRAMEWORK_FILES.json').read_bytes();assert json.loads(mf)==MANIFEST and hashlib.sha256(mf).hexdigest()==TASK['source_SHA256']['FRAMEWORK_FILES.json']==receipt['manifest_SHA256']
found={p.relative_to(root).as_posix() for p in (root/'framework').rglob('*') if p.is_file() or p.is_symlink()}
assert found==set(MANIFEST['files'])
for name,item in MANIFEST['files'].items():
    path=root/name
    assert path.is_file() and not path.is_symlink() and path.stat().st_size==item['bytes']
    assert path.stat().st_mode&0o777==item['mode']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['SHA256'],name
entry_raw=(root/'evidence/entry-verification.json').read_bytes();entry=json.loads(entry_raw)
assert hashlib.sha256(entry_raw).hexdigest()==receipt['entry_verification']['SHA256'] and entry==receipt['entry_verification']['value']
assert entry['framework_commit']==MANIFEST['framework_commit'] and len(set(entry['case_ids']))==len(entry['case_ids'])==17 and entry['python'].startswith('3.10.12 ')
entry_epoch=datetime.datetime.fromisoformat(entry['observed_UTC']).timestamp();assert receipt['epoch']<=entry_epoch<receipt['deadline']
parent=subprocess.check_output(['squeue','-h','-j','1087656','-o','%i|%T|%D|%l|%L|%N'],text=True,timeout=15).strip().split('|')
assert parent[:3]==['1087656','RUNNING','40']
nodes=subprocess.check_output(['scontrol','show','hostnames',parent[-1]],text=True,timeout=15).splitlines()
assert sorted(nodes)==TASK['expected_nodes']==sorted(receipt['actual_nodes']) and len(nodes)==len(set(nodes))==40
out={'observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'COMPLETE_UNIFIED_SOURCE_TREE_AND_TARGET_ENTRY_VERIFIED','claim_commit':receipt['claim_commit'],'receipt_SHA256':hashlib.sha256(raw).hexdigest(),'manifest_SHA256':hashlib.sha256(mf).hexdigest(),'framework_commit':MANIFEST['framework_commit'],'framework_tree':MANIFEST['framework_tree'],'source_files_rehashed':len(found),'source_bytes_rehashed':sum(x['bytes'] for x in MANIFEST['files'].values()),'source_directory':str(root/'framework'),'entry':entry,'entry_verification_SHA256':hashlib.sha256(entry_raw).hexdigest(),'actual_allocation':{'parent':parent[0],'state':parent[1],'nodes':int(parent[2]),'time_limit':parent[3],'time_left':parent[4],'node_names':nodes},'GPU_allocation_by_verifier':0,'all17_training_qualified':False}
print(json.dumps(out))
'''
r=subprocess.run(ns['ssh'],input=('TASK='+repr(task)+'\nMANIFEST='+repr(manifest)+'\n'+code).encode(),capture_output=True,timeout=60)
assert r.returncode==0,r.stderr.decode(errors='replace')[-1000:]
v=json.loads(r.stdout)
with (BASE/'independent-original-install.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps({k:x for k,x in v.items() if k not in ('entry','actual_allocation')}))
print(json.dumps({k:x for k,x in v['actual_allocation'].items() if k!='node_names'}))
