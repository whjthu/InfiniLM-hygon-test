"""Read original reader bytes and actual owners without reopening tensor data."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
transport=json.loads((D/'transport-outcome.json').read_bytes());value=transport['value'];assert transport['transport_rc']==0 and value['passed']
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',transport['claim']+':coordination/tasks/training-qwen-A100-reference-root-reader-v1.json']))
expected=dict(claim=transport['claim'],request_id=transport['request_id'],epoch=value['epoch'],deadline=value['deadline'],owners=[value['owner'],value['reader_owner']],installed={n:task['source_SHA256'][n] for n in task['remote_files']},outputs={n:dict(bytes=value[n]['bytes'],SHA256=value[n]['SHA256']) for n in ['independent-reference.json','independent-post.json','audit-progress.json']})
remote=('EXPECTED='+repr(expected)+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,os,sys
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-qwen-A100-reference-root-reader-v1')
def read(name):
 p=W/name;assert 0<p.stat().st_size<=2*1024**2;raw=p.read_bytes();return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
intent,ip=read('intent.json');outcome,op=read('outcome.json')
for k in ('claim','request_id','epoch','deadline'):assert intent[k]==outcome[k]==EXPECTED[k]
assert outcome['passed'] and outcome['reader_returncode']==0 and outcome['error_type'] is None and outcome['forced_cleanup'] is False
assert outcome['deadline']-outcome['epoch']==660 and outcome['GPU_cards']==0 and outcome['raw_tensor_transfer_bytes']==0
assert len(outcome['CPU_affinity'])==4 and outcome['peak_child_RSS_bytes']<=8*1024**3
files={}
for n,pin in EXPECTED['outputs'].items():
 row,p=read(n);assert p==pin=={k:outcome[n][k] for k in ('bytes','SHA256')};files[n]=p
for n,pin in EXPECTED['installed'].items():assert hashlib.sha256((W/'src'/n).read_bytes()).hexdigest()==pin
owners=[]
for saved in EXPECTED['owners']:
 try:
  fields=Path('/proc/%d/stat'%saved['PID']).read_text().split(') ',1)[1].split();ticks=int(fields[19]);state=fields[0]
 except FileNotFoundError:ticks=None;state=None
 assert ticks!=saved['startticks']
 owners.append(dict(PID=saved['PID'],original_startticks=saved['startticks'],current_startticks=ticks,original_identity_present=False))
group=[]
for name in os.listdir('/proc'):
 if not name.isdigit():continue
 try:fields=Path('/proc/'+name+'/stat').read_text().split(') ',1)[1].split()
 except (FileNotFoundError,ProcessLookupError):continue
 if int(fields[2])==EXPECTED['owners'][1]['PGID'] and fields[0]!='Z':group.append(int(name))
assert not group and 'torch' not in sys.modules
reference,_=read('independent-reference.json');assert reference['tensor_artifacts']==200 and reference['original25_completed_ranks']==8
assert [x['global_step'] for x in reference['checkpoints']]==[10,20,25]
print(json.dumps(dict(passed=True,at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_files={'intent':ip,'outcome':op,**files},installed_files=len(EXPECTED['installed']),installed_source_SHA_verified=True,owners=owners,reader_group_empty=True,CPU_affinity=outcome['CPU_affinity'],peak_child_RSS_bytes=outcome['peak_child_RSS_bytes'],original_reader_returncode=0,reader_Torch_imports=0,GPU_cards=0,tensor_values_reopened=False)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=45)
for name,raw in [('closure.stdout.private.json',r.stdout),('closure.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
result=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:result['value']=json.loads(r.stdout)
with (D/'reader-closure.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
v=result.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),reader_group_empty=v.get('reader_group_empty'),installed_files=v.get('installed_files'),tensor_values_reopened=False,GPU_cards=0)))
