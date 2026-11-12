"""Consume the original new CPU26 terminal/source/test log and owner exit."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
o=json.loads((D/'transport-outcome.json').read_bytes());assert o['transport_rc']==0 and o['remote_exec_acknowledged'] and o['value']['passed']
claim=o['claim'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/'+D.name+'.json']))
remote=('TASK='+repr(task)+'\nCLAIM='+repr(claim)+'\nREQUEST='+repr(o['request_id'])+'\nCPU_SHA='+repr(o['value']['CPU_result_SHA256'])+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,os,socket,sys,time
W=Path(TASK['original_namespace']);pins={}
def read(name,decode=True):
 p=W/name;a=p.stat();assert not p.is_symlink() and 0<a.st_size<=8*1024**2;raw=p.read_bytes();b=p.stat();assert (a.st_size,a.st_mtime_ns,a.st_ino,a.st_dev)==(b.st_size,b.st_mtime_ns,b.st_ino,b.st_dev)
 pins[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());return json.loads(raw) if decode else raw
intent=read('intent.json');out=read('outcome.json');cpu=read('CPU-result.json');fm=read('src/FRAMEWORK_FILES.json');tests=read('tests.private.txt',False)
assert socket.gethostname()==out['owner']['host']==intent['owner']['host']
for v in (intent,out):assert v['claim']==CLAIM and v['request_id']==REQUEST
assert intent['epoch']==out['epoch']==cpu['epoch'] and intent['deadline']==out['deadline']==cpu['epoch']+300 and cpu['deadline']==cpu['epoch']+260
assert out['passed'] and out['returncode']==0 and out['owned_group_reaped'] and out['stop_reason'] is None and out['CPU_result']==cpu
assert out['CPU_result_SHA256']==pins['CPU-result.json']['SHA256']==CPU_SHA
assert cpu['passed'] and cpu['actual_tests']==TASK['expected_Torch_CPU_tests']==26 and cpu['new_hook_tests']==8 and cpu['affected_resume_tests']==18 and cpu['new_resume_integration_tests']==2
assert cpu['framework_commit']==TASK['framework_commit']==fm['framework_commit'] and cpu['framework_files']==353 and len(fm['files'])==353
assert cpu['Torch_imported'] and not cpu['GPU_initialized'] and cpu['GPU_cards']==0 and not cpu['real_model_asset_open_attempts'] and not cpu['real_model_loaded'] and not cpu['HIP_target_qualified']
assert cpu['wall_seconds']<260 and out['wall_seconds']<300 and len(out['CPU_affinity'])==2 and out['peak_group_RSS_bytes']<=4*1024**3
assert hashlib.sha256(tests).hexdigest()==cpu['tests_SHA256']
text=tests.decode();ok=[r for r in text.splitlines() if r.startswith('test_') and r.endswith(' ... ok')]
assert len(ok)==26 and sum('BoundaryHookTest.' in r for r in ok)==8 and sum('ResumeWitnessTest.' in r for r in ok)==18 and 'Ran 26 tests' in text and text.rstrip().endswith('OK')
for remote_name,source_name in TASK['remote_files'].items():
 raw=(W/remote_name).read_bytes();assert hashlib.sha256(raw).hexdigest()==TASK['source_SHA256'][source_name]
for name,pin in fm['files'].items():
 p=W/'framework'/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
owners={}
for label,pin in [('supervisor',out['owner']),('child',out['child']),('CPU',cpu['owner'])]:
 try:current=int(Path('/proc/%d/stat'%pin['PID']).read_text().split(') ',1)[1].split()[19])
 except (FileNotFoundError,ProcessLookupError):current=None
 owners[label]=dict(PID=pin['PID'],original_startticks=pin['startticks'],current_startticks=current,original_identity_present=current==pin['startticks']);assert current!=pin['startticks']
remaining=[]
for name in os.listdir('/proc'):
 if name.isdigit():
  try:f=Path('/proc/'+name+'/stat').read_text().split(') ',1)[1].split()
  except (FileNotFoundError,ProcessLookupError):continue
  if int(f[2])==out['child']['PGID'] and f[0]!='Z':remaining.append(int(name))
assert not remaining and 'torch' not in sys.modules
print(json.dumps(dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=CLAIM,request_id=REQUEST,original_files=pins,original_CPU_result=cpu,framework_files_independently_verified=353,actual_tests=26,test_methods=ok,owners=owners,owned_group_empty=True,CPU_wall_seconds=cpu['wall_seconds'],whole_wall_seconds=out['wall_seconds'],peak_RSS_bytes=cpu['peak_RSS_bytes'],peak_sampled_group_RSS_bytes=out['peak_group_RSS_bytes'],independent_reader_Torch_imported=False,original_tests_replayed=False,real_model_loads=0,GPU_cards=0,HIP_target_qualified=False,GPU_GO=False,Model_GO=False)))
'''
with (D/'root-reader-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=o['request_id'],payload_SHA256=hashlib.sha256(remote).hexdigest(),read_only=True,tests_replayed=False),f,indent=2)
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
for n,raw in [('root-reader.stdout.private.json',r.stdout),('root-reader.stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(raw)
v=dict(transport_rc=r.returncode,script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),payload_SHA256=hashlib.sha256(remote).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:v['value']=json.loads(r.stdout)
with (D/'root-reader-result.json').open('x') as f:json.dump(v,f,indent=2)
value=v.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=value.get('passed'),actual_tests=value.get('actual_tests'),framework_files=value.get('framework_files_independently_verified'),owned_group_empty=value.get('owned_group_empty'),CPU_wall_seconds=value.get('CPU_wall_seconds'),GPU_cards=0)))
