"""Read the actual failed reference terminal/logs privately; never replay it."""
from pathlib import Path
import base64,hashlib,json,subprocess
D=Path(__file__).resolve().parent
original=json.loads((D/'transport-outcome.json').read_bytes());assert original['remote_outcome']=='ORIGINAL_RETURNED'
request=original['request_id'];claim=original['claim']
remote=('REQUEST='+repr(request)+'\nCLAIM='+repr(claim)+'\n').encode()+b'''
from pathlib import Path
import base64,datetime,hashlib,json,subprocess
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-qwen-A100-first-updates-v1')
intent=json.loads((W/'installed.json').read_bytes());assert intent['request_id']==REQUEST and intent['claim']==CLAIM
files={}
for name in ('install-intent.json','installed.json','supervisor-intent.json','gpu-intent.json','child-owner.json','owned-registry.json','outcome.json','outer-result.json','source-CPU-tests.json','native.private.log','supervisor.private.log'):
 p=W/name
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<=4*1024**2
  files[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
captures=[]
for rank in range(8):
 root=W/'capture'/('rank-%02d'%rank);row=dict(rank=rank)
 for name in ('owner.json','frozen-boundaries.json','progress.json','failure.json','receipt.json','native25-complete.json'):
  p=root/name
  if p.is_file():
   raw=p.read_bytes();assert len(raw)<2*1024**2;row[name]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
 captures.append(row)
post={}
for label,args in (
 ('GPUs',['--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
 ('apps',['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])):
 r=subprocess.run(['nvidia-smi',*args],capture_output=True,text=True,timeout=12);post[label]=dict(rc=r.returncode,rows=r.stdout.strip().splitlines())
out=json.loads((W/'outcome.json').read_bytes());owners={}
for key,pin in out['owners'].items():
 try:current=int(Path('/proc/%d/stat'%int(key)).read_text().split(') ',1)[1].split()[19])
 except FileNotFoundError:current=None
 owners[key]=dict(original_startticks=pin['startticks'],current_startticks=current,original_identity_present=current==pin['startticks'])
print(json.dumps(dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,captures=captures,post=post,owners=owners)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
with (D/'failure-read.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'failure-read.stderr.private.txt').open('xb') as f:f.write(r.stderr)
receipt=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:
 row=json.loads(r.stdout);target=D/'original-failure-files';target.mkdir(exist_ok=False)
 for name,pin in row['files'].items():
  raw=base64.b64decode(pin.pop('base64'),validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
  with (target/name).open('xb') as f:f.write(raw)
 receipt['value']=row
with (D/'failure-read-result.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,files=len(v.get('files',{})),capture_ranks_with_progress=sum('progress.json' in x for x in v.get('captures',[])),owned_original_PIDs_present=sum(x['original_identity_present'] for x in v.get('owners',{}).values()),compute_app_rows=len(v.get('post',{}).get('apps',{}).get('rows',[])))))
