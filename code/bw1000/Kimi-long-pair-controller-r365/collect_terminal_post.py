from pathlib import Path
import subprocess,json,hashlib,datetime
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run='longpair-20261102-d';rows=[]
for p in [R/'results'/('Kimi-long-pair-r347-'+run+'-watch.json')]+[R/'results'/('Kimi-long-pair-r346-'+run+'-node%d.json'%n)for n in range(4)]+[R/'results'/('Kimi-long-draft-r334-'+run+'.json')]:
 raw=p.read_bytes();rows.append(dict(path=str(p),SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),report=json.loads(raw)))
assert rows[0]['report']['completed']and rows[0]['report']['actual_target']=='910380.550'and rows[0]['report']['actual_draft']=='910371.672'
probe=r'''from pathlib import Path
import os,socket,json,datetime
owned=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit()or int(p.name)==os.getpid():continue
 try:args=(p/'cmdline').read_bytes().split(b'\0')
 except(OSError,PermissionError):continue
 scripts=[a for a in args if a.startswith(b'/refactor/incoming/Kimi-long-')and a.endswith((b'/worker.py',b'/bootstrap.py',b'/pressure.py'))]
 if scripts:owned.append({'PID':int(p.name),'entry':Path(os.fsdecode(scripts[0])).name})
kfd=sorted(int(x.name)for x in Path('/sys/class/kfd/kfd/proc').iterdir()if x.name.isdigit());ports={};entries=[]
for port in [32947,39947,45947,46947]:
 with socket.socket()as s:
  try:s.bind(('0.0.0.0',port));ports[str(port)]=True
  except OSError:ports[str(port)]=False
for table in ['/proc/net/tcp','/proc/net/tcp6']:
 for line in Path(table).read_text().splitlines()[1:]:
  f=line.split();port=int(f[1].split(':')[-1],16)
  if port in [32947,39947,45947,46947]:entries.append({'port':port,'state_hex':f[3],'inode':f[9]})
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname().split('.')[0],'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,'GPU_initialized':False,'TCP_connections_made':0,'owned_long_worker_PIDs':owned,'KFD_pids':kfd,'ports_bindable':ports,'TCP_entries':entries,'passed':not owned and not kfd and all(ports.values())}))
'''
post=[]
for parent,nodes in [(910380,['a14r2n08','a14r2n09','a14r2n10','a14r2n12']),(910371,['a02r3n15'])]:
 args=['srun','--jobid='+str(parent),'--overlap','--exact','--immediate=30','--cpu-bind=none','-N'+str(len(nodes)),'-n'+str(len(nodes)),'--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(nodes),'-t','00:03:00','--job-name=Kimi-run-d-independent-post','python3','-B','-c',probe]
 p=subprocess.run(args,text=True,capture_output=True,timeout=150);assert p.returncode==0,p.stderr
 reports=[json.loads(x)for x in p.stdout.splitlines()if x.startswith('{')];assert {x['node']for x in reports}==set(nodes);post.extend(reports)
account=subprocess.check_output(['sacct','-S','2026-11-01','-j','910380.550,910371.672,910407.719,910380.552,910380.553,910371.676,910371.677','-n','-P','-o','JobIDRaw,State,ExitCode,Elapsed,NodeList%120'],text=True,timeout=30)
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run,GPU_new=0,files=rows,independent_post=post,all5_independent_post_clean=all(x['passed']for x in post),accounting=account)))
