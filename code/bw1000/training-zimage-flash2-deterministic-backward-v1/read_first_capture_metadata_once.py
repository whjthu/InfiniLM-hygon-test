"""Bind the newly completed candidate first-update reference;metadata only."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');i=json.loads((D/'launch-intent.json').read_bytes())
code=('CLAIM='+repr(i['claim'])+'\nREQUEST='+repr(i['request_id'])+'\n').encode()+br'''
from pathlib import Path
import hashlib,json,signal
signal.alarm(20);w=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-flash2-deterministic-backward-v1')
def read(p):
 b=p.read_bytes();assert not p.is_symlink() and len(b)<3*1024**2
 return json.loads(b),dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())
i,_=read(w/'intent.json');assert i['claim']==CLAIM and i['request_id']==REQUEST
r,_=read(w/'candidate-continuous25/result.json');m,mp=read(w/'candidate-continuous25/first-update-private/progress.json');b,bp=read(w/'candidate-continuous25/first-update-private/binding.json')
cfg,_=read(w/'src/BINDING.json')
assert r['passed'] and r['completed_global_steps']==list(range(1,26)) and r['first_update_capture_complete']
assert m['phase']=='NATIVE25_RETURNED' and m['completed_global_steps']==25 and m['first_update_complete'] and m['trainable_tensors']==476 and m['trainable_elements']==9922560 and len(m['files'])==12
assert b['config_SHA256']==cfg['config_canonical_SHA256'] and cfg['config']['attention_backward_policy']=='flash2-deterministic-backward-v1'
for pin in m['files']:
 p=w/'candidate-continuous25/first-update-private'/pin['name'];assert not p.is_symlink() and p.stat().st_size==pin['bytes']
assert sum(p['bytes'] for p in m['files'])==m['artifact_bytes']==r['first_update_capture_bytes']
print(json.dumps(dict(passed=True,source_commit=CLAIM,request_id=REQUEST,relative_manifest='candidate-continuous25/first-update-private/progress.json',manifest_pin=mp,binding_pin=bp,
 config_SHA256=b['config_SHA256'],capture_files=m['files'],capture_bytes=m['artifact_bytes'],actual_trainable_tensors=476,actual_trainable_elements=9922560,actual_first_loss=r['step_metrics'][0]['loss'],
 raw_tensor_bytes_read=0,new_model_loads=0,new_GPU_runs=0,new_platform_alignment_qualified=False)))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -'],input=code,capture_output=True,timeout=35)
with (D/'first-capture-metadata.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'first-capture-metadata.stderr.private.txt').open('xb') as f:f.write(r.stderr)
v=json.loads(r.stdout) if r.stdout.strip() else None
with (D/'first-capture-metadata-result.json').open('x') as f:json.dump(dict(transport_rc=r.returncode,value=v,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest()),f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed') if v else None,capture_bytes=v.get('capture_bytes') if v else None,raw_tensor_bytes_read=0,new_GPU_runs=0)))
