"""Independently consume only the original terminal, metadata and header bytes."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
original=json.loads((D/'transport-outcome.json').read_bytes());assert original['transport_rc']==0 and original['remote_exec_acknowledged'] and original['value']['passed']
claim=original['claim'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/'+D.name+'.json']))
cpu_sha=original['value']['CPU_result_SHA256'];identity_sha=original['value']['MODEL_IDENTITY.json']['SHA256']
remote=('TASK='+repr(task)+'\nCLAIM='+repr(claim)+'\nREQUEST='+repr(original['request_id'])+'\nCPU_SHA='+repr(cpu_sha)+'\nIDENTITY_SHA='+repr(identity_sha)+'\n').encode()+b'''
from pathlib import Path
from collections import Counter
import datetime,hashlib,json,math,os,socket,sys,time
start=time.monotonic();W=Path(TASK['original_namespace']);pins={}
def read(name):
 p=W/name;a=p.stat();assert not p.is_symlink() and 0<a.st_size<=2*1024**2;raw=p.read_bytes();b=p.stat();assert (a.st_size,a.st_mtime_ns,a.st_ino,a.st_dev)==(b.st_size,b.st_mtime_ns,b.st_ino,b.st_dev)
 pins[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());return json.loads(raw)
intent=read('intent.json');out=read('outcome.json');cpu=read('CPU-result.json');model=read('MODEL_IDENTITY.json');progress=read('hash-progress.json');expected=read('src/ORIGINAL_MODEL_METADATA.json')
assert socket.gethostname()==intent['owner']['host']==expected['host']
for doc in (intent,out):assert doc['claim']==CLAIM and doc['request_id']==REQUEST
assert intent['epoch']==out['epoch']==cpu['epoch']==model['epoch'] and intent['deadline']==out['deadline']==cpu['epoch']+300 and cpu['deadline']==model['deadline']==cpu['epoch']+260
assert out['passed'] and out['returncode']==0 and out['owned_group_reaped'] and out['stop_reason'] is None
assert out['CPU_result']==cpu and pins['CPU-result.json']['SHA256']==out['CPU_result_SHA256']==CPU_SHA
assert out['MODEL_IDENTITY.json']['value']==model and pins['MODEL_IDENTITY.json']['SHA256']==out['MODEL_IDENTITY.json']['SHA256']==cpu['MODEL_IDENTITY_SHA256']==IDENTITY_SHA
assert cpu['passed'] and cpu['framework_commit']==TASK['framework_commit'] and cpu['framework_subset_files']==1 and cpu['new_stdlib_fixture_runs']==0
assert cpu['original_shards']==4 and cpu['tensors']==model['tensors']==698
assert cpu['weight_bytes_read_for_hash'] and not cpu['tensor_values_decoded'] and not cpu['weight_values_exported'] and not cpu['model_loaded'] and not cpu['Torch_imported'] and cpu['GPU_cards']==0
assert cpu['wall_seconds']<260 and out['wall_seconds']<300 and len(out['CPU_affinity'])==2 and out['peak_group_RSS_bytes']<=1024**3
assert pins['src/ORIGINAL_MODEL_METADATA.json']['SHA256']==model['original_metadata_SHA256']
for remote_name,source_name in TASK['remote_files'].items():
 raw=(W/remote_name).read_bytes();assert hashlib.sha256(raw).hexdigest()==TASK['source_SHA256'][source_name]
raw=(W/'hash-progress.private.jsonl').read_bytes();assert len(raw)<=2*1024**2 and hashlib.sha256(raw).hexdigest()==cpu['hash_progress_SHA256']
events=[json.loads(line) for line in raw.splitlines()]
assert len(events)==cpu['progress_events'] and events[-1]==progress and events[0]['stage']=='index_verified' and events[0]['bytes_read']==0
assert all(len(e['planned_shards'])==4 and set(e['planned_shards'])==set(model['shards']) and set(e['completed_shards'])<=set(model['shards']) for e in events)
assert all(a['bytes_read']<=b['bytes_read'] for a,b in zip(events,events[1:]))
assert progress['stage']=='shard_complete' and set(progress['completed_shards'])==set(progress['planned_shards'])==set(model['shards'])
assert progress['bytes_read']==cpu['full_file_bytes_read']==model['full_file_bytes_read']==TASK['declared_full_file_bytes']==17054506776
assert not model['native_loaded_schema_correspondence_qualified'] and not model['official_release_provenance_qualified'] and not model['historical_training_file_bytes_qualified'] and not model['actual_weight_loading_qualified']
root=Path(model['model_root']);index=root/model['index']['name'];raw=index.read_bytes()
assert hashlib.sha256(raw).hexdigest()==model['index']['SHA256']==expected['files']['index']['SHA256'] and len(raw)==model['index']['bytes']
index=json.loads(raw);assert len(index['weight_map'])==698
config=model['config'];p=Path(config['path']);assert p.stat().st_size==config['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==config['SHA256']
seen=set();payload=elements=header_reads=0;counts=Counter();shards={}
sizes={'BOOL':1,'U8':1,'I8':1,'I16':2,'I32':4,'I64':8,'F16':2,'BF16':2,'F32':4,'F64':8}
declared={p['name']:p for p in expected['shards']}
for name,pin in model['shards'].items():
 p=root/name;s=p.stat();assert not p.is_symlink()
 current=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
 assert current=={k:pin[k] for k in current}=={k:declared[name][k] for k in current}
 assert len(pin['SHA256'])==64 and all(c in '0123456789abcdef' for c in pin['SHA256'])
 with p.open('rb') as f:
  lead=f.read(8);length=int.from_bytes(lead,'little');assert length==pin['header_bytes'] and 0<length<=8*1024**2;header=f.read(length)
 assert len(header)==length and hashlib.sha256(header).hexdigest()==pin['header_SHA256'];header_reads+=8+length
 after=p.stat();assert current==dict(bytes=after.st_size,mtime_ns=after.st_mtime_ns,inode=after.st_ino,device=after.st_dev)
 raw_tensors={k:v for k,v in json.loads(header).items() if k!='__metadata__'};assert set(raw_tensors)==set(pin['tensors'])
 cursor=0
 for key,v in sorted(pin['tensors'].items(),key=lambda pair:pair[1]['data_offsets']):
  assert key not in seen and index['weight_map'][key]==name;seen.add(key)
  h=raw_tensors[key];assert all(h[k]==v[k] for k in ('shape','dtype','data_offsets'))
  lo,hi=v['data_offsets'];n=math.prod(v['shape']);assert lo==cursor and hi-lo==n*sizes[v['dtype']]==v['bytes'];cursor=hi
  assert v['elements']==n;payload+=hi-lo;elements+=n;counts[v['dtype']]+=1
 assert cursor+pin['header_bytes']+8==pin['bytes']
 shards[name]={k:pin[k] for k in ('bytes','mtime_ns','inode','device','SHA256','header_bytes','header_SHA256')};shards[name]['tensors']=len(pin['tensors'])
assert seen==set(index['weight_map']) and payload==cpu['tensor_payload_bytes']==model['tensor_payload_bytes']==index['metadata']['total_size']==TASK['declared_tensor_payload_bytes']==17054429248
assert elements==cpu['serialized_tensor_elements']==model['serialized_tensor_elements']
owners={}
for label,pin in [('supervisor',out['owner']),('child',out['child']),('CPU',cpu['owner'])]:
 try:current=int(Path('/proc/%d/stat'%pin['PID']).read_text().split(') ',1)[1].split()[19])
 except (FileNotFoundError,ProcessLookupError):current=None
 owners[label]=dict(PID=pin['PID'],original_startticks=pin['startticks'],current_startticks=current,original_identity_present=current==pin['startticks']);assert current!=pin['startticks']
group=[]
for name in os.listdir('/proc'):
 if name.isdigit():
  try:f=Path('/proc/'+name+'/stat').read_text().split(') ',1)[1].split()
  except (FileNotFoundError,ProcessLookupError):continue
  if int(f[2])==out['child']['PGID'] and f[0]!='Z':group.append(int(name))
assert not group and 'torch' not in sys.modules
print(json.dumps(dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=CLAIM,request_id=REQUEST,original_files=pins,
 installed_files_verified=len(TASK['remote_files']),shards=shards,tensors=698,dtype_tensor_counts=dict(counts),full_file_bytes_read_by_original=cpu['full_file_bytes_read'],
 independent_payload_rehash=False,independent_header_bytes_read=header_reads,independent_header_identity_passed=True,tensor_payload_bytes=payload,serialized_tensor_elements=elements,progress_events=len(events),
 owners=owners,owned_group_empty=True,wall_seconds=out['wall_seconds'],CPU_wall_seconds=cpu['wall_seconds'],hash_wall_seconds=cpu['hash_wall_seconds'],
 peak_RSS_bytes=cpu['peak_RSS_bytes'],peak_sampled_group_RSS_bytes=out['peak_group_RSS_bytes'],framework_commit=cpu['framework_commit'],framework_subset_files=1,
 Torch_imported=False,GPU_cards=0,weight_bytes_read_for_hash=True,tensor_values_decoded=False,weight_values_exported=False,actual_model_loaded=False,
 native_loaded_schema_correspondence_qualified=False,official_release_provenance_qualified=False,first_update_numerics_qualified=False,GPU_GO=False,Model_GO=False,reader_wall_seconds=time.monotonic()-start)))
'''
with (D/'root-reader-intent.json').open('x') as f:json.dump(dict(payload_SHA256=hashlib.sha256(remote).hexdigest(),claim=claim,request_id=original['request_id'],read_only=True,full_payload_rehash=False),f,indent=2)
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
for name,raw in [('root-reader.stdout.private.json',r.stdout),('root-reader.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
receipt=dict(transport_rc=r.returncode,script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),payload_SHA256=hashlib.sha256(remote).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
with (D/'root-reader-result.json').open('x') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),tensors=v.get('tensors'),full_file_bytes_read_by_original=v.get('full_file_bytes_read_by_original'),dtype_counts=v.get('dtype_tensor_counts'),owned_group_empty=v.get('owned_group_empty'),GPU_cards=0)))
