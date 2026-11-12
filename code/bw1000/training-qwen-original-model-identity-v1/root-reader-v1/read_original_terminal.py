"""Consume the original full-hash terminal and independently inspect headers/stats."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
original=json.loads((D/'transport-outcome.json').read_bytes());assert original['transport_rc']==0 and original['value']['passed']
claim=original['claim'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/training-qwen-original-model-identity-v1.json']))
expected=dict(claim=claim,request_id=original['request_id'],epoch=original['value']['epoch'],deadline=original['value']['deadline'],CPU_result_SHA256=original['value']['CPU_result_SHA256'])
pins={remote:task['source_SHA256'][source] for remote,source in task['remote_files'].items()}
remote=('EXPECTED='+repr(expected)+'\nPINS='+repr(pins)+'\n').encode()+b'''
from pathlib import Path
from collections import Counter
import datetime,hashlib,json,math,os,sys,time
start=time.monotonic();W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-qwen-original-model-identity-v1')
def read(name):
 p=W/name;a=p.stat();assert 0<a.st_size<=2*1024**2;raw=p.read_bytes();b=p.stat();assert(a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
def unique(pairs):
 value={}
 for k,v in pairs:
  assert k not in value;value[k]=v
 return value
intent,ip=read('intent.json');outcome,op=read('outcome.json');cpu,cp=read('CPU-result.json');model,mp=read('MODEL_IDENTITY.json');progress,pp=read('hash-progress.json')
for name in ('claim','request_id','epoch','deadline'):assert intent[name]==outcome[name]==EXPECTED[name]
assert cp['SHA256']==outcome['CPU_result_SHA256']==EXPECTED['CPU_result_SHA256']
assert outcome['passed'] and outcome['returncode']==0 and outcome['owned_group_reaped'] and outcome['stop_reason'] is None
assert math.isclose(outcome['deadline']-outcome['epoch'],300,abs_tol=1e-5) and cpu['epoch']==outcome['epoch']
assert math.isclose(cpu['deadline']-cpu['epoch'],260,abs_tol=1e-5)
assert cpu['passed'] and cpu['shared_tests']==6 and cpu['original_shards']==4 and cpu['tensors']==750
assert cpu['MODEL_IDENTITY_SHA256']==mp['SHA256']==outcome['MODEL_IDENTITY.json']['SHA256'] and outcome['MODEL_IDENTITY.json']['value']==model
assert cpu['weight_bytes_read_for_hash'] and not cpu['tensor_values_decoded'] and not cpu['model_loaded'] and not cpu['Torch_imported'] and cpu['GPU_cards']==0
assert progress['stage']=='shard_complete' and set(progress['completed_shards'])==set(progress['planned_shards'])==set(model['shards'])
assert progress['bytes_read']==cpu['full_file_bytes_read']==model['full_file_bytes_read']==17534339512
for name,pin in PINS.items():
 raw=(W/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==pin
raw=(W/'hash-progress.private.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==cpu['hash_progress_SHA256']
events=[json.loads(line) for line in raw.splitlines()];assert len(events)==cpu['progress_events'] and events[-1]==progress
assert all(len(event['planned_shards'])==4 and set(event['completed_shards'])<=set(model['shards']) for event in events)
assert all(a['bytes_read']<=b['bytes_read'] for a,b in zip(events,events[1:]))
config=model['config'];p=Path(config['path']);assert p.stat().st_size==config['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==config['SHA256']
root=Path(model['model_root']);index=root/model['index']['name'];raw=index.read_bytes();assert len(raw)==model['index']['bytes'] and hashlib.sha256(raw).hexdigest()==model['index']['SHA256']
document=json.loads(raw,object_pairs_hook=unique);weight_map=document['weight_map'];assert len(weight_map)==750
counts=Counter();payload=elements=0;seen=set();headers={}
unit={'BOOL':1,'U8':1,'I8':1,'I16':2,'I32':4,'I64':8,'F16':2,'BF16':2,'F32':4,'F64':8}
for name,pin in model['shards'].items():
 p=root/name;a=p.stat();assert not p.is_symlink()
 assert dict(bytes=a.st_size,mtime_ns=a.st_mtime_ns,inode=a.st_ino,device=a.st_dev)=={k:pin[k] for k in ('bytes','mtime_ns','inode','device')}
 assert len(pin['SHA256'])==64 and set(pin['SHA256'])<=set('0123456789abcdef')
 with p.open('rb') as f:
  n=int.from_bytes(f.read(8),'little');assert n==pin['header_bytes'] and 0<n<=8*1024**2
  raw=f.read(n)
 assert hashlib.sha256(raw).hexdigest()==pin['header_SHA256'];header=json.loads(raw,object_pairs_hook=unique);header.pop('__metadata__',None)
 assert set(header)==set(pin['tensors'])=={k for k,v in weight_map.items() if v==name}
 spans=[]
 for key,v in header.items():
  assert key not in seen;seen.add(key);assert v['dtype'] in unit and all(type(d) is int and d>=0 for d in v['shape'])
  count=math.prod(v['shape']);lo,hi=v['data_offsets'];assert hi-lo==count*unit[v['dtype']]
  assert pin['tensors'][key]==dict(dtype=v['dtype'],shape=v['shape'],data_offsets=v['data_offsets'],elements=count,bytes=hi-lo)
  spans.append((lo,hi));payload+=hi-lo;elements+=count;counts[v['dtype']]+=1
 cursor=0
 for lo,hi in sorted(spans):assert lo==cursor and hi>=lo;cursor=hi
 assert cursor+n+8==a.st_size and a.st_size==p.stat().st_size and a.st_mtime_ns==p.stat().st_mtime_ns
 headers[name]=dict(bytes=a.st_size,SHA256=pin['SHA256'],header_bytes=n,header_SHA256=pin['header_SHA256'],tensors=len(header))
assert seen==set(weight_map) and payload==model['tensor_payload_bytes']==document['metadata']['total_size'] and elements==model['serialized_tensor_elements']
owners={}
for label,pin in [('supervisor',outcome['owner']),('child',outcome['child']),('CPU',cpu['owner'])]:
 try:current=int(Path('/proc/%d/stat'%pin['PID']).read_text().split(') ',1)[1].split()[19])
 except FileNotFoundError:current=None
 owners[label]=dict(PID=pin['PID'],original_startticks=pin['startticks'],current_startticks=current,original_identity_present=current==pin['startticks']);assert current!=pin['startticks']
group=[]
for name in os.listdir('/proc'):
 if name.isdigit():
  try:f=Path('/proc/'+name+'/stat').read_text().split(') ',1)[1].split()
  except (FileNotFoundError,ProcessLookupError):continue
  if int(f[2])==outcome['child']['PGID'] and f[0]!='Z':group.append(int(name))
assert not group and 'torch' not in sys.modules
tests=(W/'tests.txt').read_bytes();assert b'Ran 6 tests' in tests and b'OK' in tests
print(json.dumps(dict(passed=True,at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=outcome['claim'],request_id=outcome['request_id'],
 original_files={'intent':ip,'outcome':op,'CPU_result':cp,'MODEL_IDENTITY':mp,'progress':pp},epoch=outcome['epoch'],deadline=outcome['deadline'],
 installed_files=len(PINS),shared_tests=6,shards=headers,tensors=750,dtype_tensor_counts=dict(counts),
 full_file_bytes_read_by_original=cpu['full_file_bytes_read'],independent_full_payload_rehash=False,independent_all_header_and_stat_checks=True,
 tensor_payload_bytes=payload,serialized_tensor_elements=elements,progress_events=len(events),owners=owners,owned_group_empty=True,
 wall_seconds=outcome['wall_seconds'],CPU_wall_seconds=cpu['wall_seconds'],hash_wall_seconds=cpu['hash_wall_seconds'],
 peak_RSS_bytes=cpu['peak_RSS_bytes'],peak_group_RSS_bytes=outcome['peak_group_RSS_bytes'],framework_commit=cpu['framework_commit'],
 framework_subset_files=cpu['framework_subset_files'],weight_bytes_read_for_hash=True,tensor_values_decoded=False,model_loaded=False,GPU_cards=0,GPU_GO=False,Model_GO=False,
 reader_wall_seconds=time.monotonic()-start)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
with (D/'root-reader-intent.json').open('x',encoding='utf8') as f:json.dump(dict(payload_SHA256=hashlib.sha256(remote).hexdigest(),claim=claim,request_id=original['request_id'],read_only=True,full_payload_rehash=False),f,indent=2)
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
with (D/'root-reader.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'root-reader.stderr.private.txt').open('xb') as f:f.write(r.stderr)
receipt=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
with (D/'root-reader-result.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),tensors=v.get('tensors'),dtype_tensor_counts=v.get('dtype_tensor_counts'),owned_group_empty=v.get('owned_group_empty'),GPU_cards=0)))
