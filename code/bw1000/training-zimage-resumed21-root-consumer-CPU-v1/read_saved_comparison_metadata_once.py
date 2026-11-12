"""Read existing comparisonJSON and audit original/newCPU metadata only."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;N=D.parent/'training-zimage-resumed21-native-boundary-pair-v1';A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
i=json.loads((D/'launch-intent.json').read_bytes());old=json.loads((N/'launch-intent.json').read_bytes())
binding=json.loads((D/'publication/code/bw1000'/D.name/'BINDING.json').read_bytes())
code=('CPU_CLAIM='+repr(i['claim'])+'\nCPU_REQUEST='+repr(i['request_id'])+'\nOLD_CLAIM='+repr(old['claim'])+'\nOLD_REQUEST='+repr(old['request_id'])+'\nEXPECTED_OLD_FILES='+repr(binding['original_remote_source_pins'])+'\n').encode()+br'''
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess
signal.alarm(45);w=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-resumed21-native-boundary-pair-v1');c=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-resumed21-root-consumer-CPU-v1')
artifacts={}
def read(root,n):
 p=root/n;assert not p.is_symlink() and p.is_file() and p.stat().st_size<2*1024**2
 b=p.read_bytes();v=json.loads(b);artifacts[('GPU/' if root==w else 'CPU/')+n]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),value=v);return v
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gone(owner):
 try:
  f=Path('/proc/%d/stat'%owner['PID']).read_text().split(') ',1)[1].split();assert int(f[19])!=owner['startticks'] or f[0]=='Z'
 except (FileNotFoundError,ProcessLookupError):pass
ci=read(c,'intent.json');co=read(c,'outcome.json');cr=read(c,'CPU-result.json')
assert ci['claim']==co['claim']==CPU_CLAIM and ci['request_id']==co['request_id']==CPU_REQUEST
assert co['returncode']==1 and co['passed'] is False and co['owned_group_reaped'] and co['stop_reason'] is None
assert cr['passed'] is False and cr['phase']=='consume-original-saved-native-arms-no-GPU' and cr['error_type']=='SystemExit'
assert cr['CPU_tests']==dict(tests=3,failures=0,errors=0,skipped=0,passed=True)
comp=read(c,'recovered-comparison.json');assert artifacts['CPU/recovered-comparison.json']['SHA256']=='2aba9ff48b38cae67782b5c60d6f270563b93701988afa45323d5a86b7e64191'
assert comp['passed'] and comp['first_different_group']=='preclip.pt' and comp['actual_optimizer_calls']==3 and comp['original_GPU_overall_passed'] is False
for n,pin in EXPECTED_OLD_FILES.items():assert (w/n).stat().st_size==pin['bytes'] and sha(w/n)==pin['SHA256']
gi=read(w,'intent.json');go=read(w,'outcome.json');gr=read(w,'GPU-result.json')
assert gi['claim']==go['claim']==OLD_CLAIM and gi['request_id']==go['request_id']==OLD_REQUEST
assert go['returncode']==1 and go['passed'] is False and go['owned_group_reaped'] and go['stop_reason'] is None
assert gr['passed'] is False and gr['stage']=='actual-private-tensor-comparison' and gr['error_type']=='ValueError'
assert 'framework_commit' not in gr
assert [p['arm'] for p in gr['phases']]==['cold','warm-restored'] and sum(p['result']['actual_optimizer_updates'] for p in gr['phases'])==3
owners=[co['owner'],co['child'],cr['owner'],go['owner'],go['child'],gr['owner']]
for p in gr['phases']:
 arm=p['arm'];v=read(w,arm+'/result.json');m=read(w,arm+'/resumed-update-private/progress.json');rb=read(w,arm+'/resumed-update-private/binding.json')
 assert p['returncode']==0 and v['passed'] and v==p['result'] and artifacts['GPU/'+arm+'/result.json']['SHA256']==p['result_SHA256']
 assert v['completed_global_steps']==[21] and v['actual_optimizer_updates']==(1 if arm=='cold' else 2) and v['first_update_capture_complete'] and v['frozen_parameters_unchanged']
 assert m['phase']=='RESUMED21_RETURNED' and m['completed_global_steps']==[21] and m['trainable_tensors']==476 and m['trainable_elements']==9922560 and len(m['files'])==12
 assert rb['checkpoint20_SHA256']=='ea76e216a4c36f6b0b21f2b5d1cc49b342ea8e7224fea0cf82ded238140099ed'
 assert m['artifact_bytes']==sum(x['bytes'] for x in m['files'])==v['first_update_capture_bytes']
 for x in m['files']:
  f=w/arm/'resumed-update-private'/x['name'];assert not f.is_symlink() and f.stat().st_size==x['bytes']
 assert sha(w/arm/'frozen-before.json')==sha(w/arm/'frozen-after.json')
 if arm=='warm-restored':assert v['diagnostic_warmup']['restored20_after_warmup']
 owners.append(p['owner'])
for root,num,commit in [(w,369,'0a999f9c7d5124f4f4b21078c10c16f6613ec39e'),(c,371,'e9d2800fadd7b879c3945374f7bc777c9bea1906')]:
 fm=json.loads((root/'src/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==num and fm['framework_commit']==commit
 for n,pin in fm['files'].items():
  f=root/'framework'/n;assert not f.is_symlink() and f.stat().st_size==pin['bytes'] and sha(f)==pin['SHA256'] and f.stat().st_mode&0o777==pin['mode']
for owner in owners:gone(owner)
for name in os.listdir('/proc'):
 if name.isdigit():
  try:
   f=Path('/proc/'+name+'/stat').read_text().split(') ',1)[1].split();assert f[0]=='Z' or int(f[2]) not in (go['child']['PGID'],co['child']['PGID'])
  except (FileNotFoundError,ProcessLookupError):pass
uuid='GPU-38411ef6-aa2e-b1f4-b72e-e9aab37cdda2';pci='00000000:65:00.0'
rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines();cards=[[v.strip() for v in r.split(',')] for r in rows if uuid in r]
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
assert len(cards)==1 and cards[0][1]==pci and int(cards[0][2])<=16 and not any(uuid in r for r in apps)
print(json.dumps(dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifacts=artifacts,post=dict(cards=cards,owned_compute_apps=[]),
 original_GPU_overall_passed=False,original_CPU_consumer_overall_passed=False,source_files_checked=[369,371],original_native_updates=3,new_GPU_runs=0,new_CPU_tests=0,raw_tensor_bytes_read=0,Model_GO=False)))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -'],input=code,capture_output=True,timeout=60)
with (D/'saved-comparison-metadata.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'saved-comparison-metadata.stderr.private.txt').open('xb') as f:f.write(r.stderr)
v=json.loads(r.stdout) if r.stdout.strip() else None
with (D/'saved-comparison-metadata-result.json').open('x') as f:json.dump(dict(transport_rc=r.returncode,value=v,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest()),f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed') if v else None,raw_tensor_bytes_read=0,new_GPU_runs=0)))
