"""Host-only fresh five-node PID/KFD/port observation, before the invoker."""
from pathlib import Path
import ast

D=Path(__file__).resolve().parent;L=D.parents[3]
probe="""from pathlib import Path
import os,socket,json,datetime
owned=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit()or int(p.name)==os.getpid():continue
 try:args=(p/'cmdline').read_bytes().split(b'\\0')
 except(OSError,PermissionError):continue
 scripts=[a for a in args if a.startswith(b'/refactor/incoming/Kimi-long-')and a.endswith((b'/worker.py',b'/bootstrap.py',b'/pressure.py'))]
 if scripts:owned.append({'PID':int(p.name),'entry':Path(os.fsdecode(scripts[0])).name})
kfd=sorted(int(x.name)for x in Path('/sys/class/kfd/kfd/proc').iterdir()if x.name.isdigit());ports={}
for port in [32947,39947,45947,46947]:
 with socket.socket()as s:
  try:s.bind(('0.0.0.0',port));ports[str(port)]=True
  except OSError:ports[str(port)]=False
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname().split('.')[0],'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,'GPU_initialized':False,'TCP_connections_made':0,'owned_long_worker_PIDs':owned,'KFD_pids':kfd,'ports_bindable':ports,'passed':not owned and not kfd and all(ports.values())}))
"""
script="""import subprocess,json,time,datetime
rows=[];groups=[]
for parent,nodes in [(910380,['a14r2n08','a14r2n09','a14r2n10','a14r2n12']),(910371,['a02r3n15'])]:
 args=['srun','--jobid='+str(parent),'--overlap','--exact','--immediate=30','--cpu-bind=none','-N'+str(len(nodes)),'-n'+str(len(nodes)),'--ntasks-per-node=1','-c1','--gres=none','--mem=256M','-w',','.join(nodes),'-t','00:02:00','--job-name=Kimi-run-d-PID-before','python3','-B','-c',__PROBE__]
 p=subprocess.run(args,text=True,capture_output=True,timeout=150);assert p.returncode==0,p.stderr
 items=[json.loads(x)for x in p.stdout.splitlines()if x.startswith('{')];assert len(items)==len(nodes)and {x['node']for x in items}==set(nodes)
 rows.extend(items);assert all(x['passed']for x in items)
 steps={x['actual_step']for x in items};assert len(steps)==1;step=next(iter(steps));until=time.monotonic()+90
 while True:
  raw=subprocess.check_output(['sacct','-S','2026-11-01','-j',step,'-n','-P','-o','JobIDRaw,State,ExitCode'],text=True,timeout=10)
  account=[x.split('|')for x in raw.splitlines()if x.startswith(step+'|')]
  if account and account[0][1:3]==['COMPLETED','0:0']:break
  assert time.monotonic()<until,'Own freshPID probe terminal accounting unresolved; do not invokeGPU';time.sleep(1)
 groups.append({'actual_step':step,'accounting':raw})
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU':0,'CPU_only':True,'run':'longpair-20261102-d','passed':True,'nodes':rows,'probe_groups':groups,'source':'published365prepare_PID_probe; originalcontroller subsequently repeats membership/Slurm dedupe/KFD/fourport checks before originalGPUepoch'}))
""".replace('__PROBE__',repr(probe))
ast.parse(probe);ast.parse(script);(L/'remote364365-fresh-PID.py').write_bytes(script.encode())
print('Fresh five-node PID helper AST PASS; no execution')
