import datetime,hashlib,json,os,re,subprocess
from pathlib import Path
base=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
label='training-qwen-vision-sdpa-policy-pair-v1'
request='b030b26749724cbea90d7f8f31f08c6d'
r=base/label
def read(p):
 if not p.is_file():return None,None
 raw=p.read_bytes()
 assert 0<len(raw)<=1024*1024
 return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
def query(*args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=8)
 assert p.returncode==0 and len(p.stdout)<=1024*1024
 return p.stdout.splitlines()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
fmt='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160'
rows=[x for s in query('sacct','-n','-P','-j','1103606','--format='+fmt)
 if (x:=s.split('|'))[1] in (label,label+'-pre',label+'-post')]
assert len(rows)==len({x[1] for x in rows})
d,owner_pin=read(r/'keeper-owner.json')
live=None
if d:
 assert d['request_id']==request
 boot=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest()
 if d['boot_id_SHA256']==boot:
  proc=Path('/proc')/str(d['PID'])/'stat'
  try: parts=proc.read_text().split(') ',1)[1].split()
  except FileNotFoundError:live=False
  else:live=int(parts[2])==d['PGID'] and int(parts[19])==d['startticks'] and parts[0]!='Z'
result=dict(UTC=now,request=request,accounting_columns=fmt.split(','),rows=rows,namespace=r.is_dir(),
 controller_live=live,owner_pin=owner_pin,preflight=(r/'source-and-asset-preflight.json').is_file(),rank_records={},pins={})
for name in ('assigned.json','native-outcome.json','outcome.json','before.json','after.json'):
 value,pin=read(r/name)
 if value is not None:
  result['pins'][name]=pin
  if name=='assigned.json':result['assigned']=dict(step=value['step'],cards=len(set(value['PCI_by_index'])),node=value['node'])
  if name=='outcome.json':result['controller']=dict(passed=value['keeper_passed'],GPU_step_requested=value['GPU_step_requested'],
       error_class=(value['error'] or {}).get('type'),clean_post=bool(value['after'] and value['after']['owned_processes_remaining']==[]))
  if name=='native-outcome.json':result['native']=dict(normal=value['normal_native_exit'],rc=value['native_returncode'],
      rank_record_counts=value['rank_record_counts'],timed_out=value['timed_out'],cleanup={k:value['cleanup'].get(k) for k in ('termination_needed','owned_processes_remaining')})
  if name in ('before.json','after.json'):
   result[name.split('.')[0]+ '_gate']=dict(same_node=value['node'],PCI=value['pci'],boot_id=value['boot_id'],holders=value['kfd_holder_count'],step=value['step'])
for rank in range(8):
 p=r/f'repeat-rank-{rank:02d}.json'
 q,pin=read(p)
 if q is None:continue
 assert q['original_recorder_validated_and_detached'] and q['rank']==rank and q['backward_calls']==q['optimizer_calls']==0 and len(q['records'])==4
 result['rank_records'][str(rank)]=q
 result['pins'][p.name]=pin
if result.get('controller') and not result['controller']['passed']:
 for name in ('native.log','keeper.log',label+'.log'):
  p=r/name
  if p.is_file():
   with p.open('rb') as f:
    f.seek(max(0,p.stat().st_size-65536))
    lines=f.read(65536).decode(errors='replace').splitlines()
   keys=['ValueError','AssertionError','ModuleNotFoundError','ImportError','RuntimeError','AttributeError','FileNotFoundError','TypeError','KeyError','TimeoutError']
   result.setdefault('restricted_error_signatures',{})[name]=[x.strip()[:170] for x in lines if any(re.match(r'^(?:\[rank\d+\]:\s*)?'+k+r':',x.strip()) for k in keys)][-12:]
if __name__=='__main__':print(json.dumps(result,allow_nan=False))
