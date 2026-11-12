"""One finite OFF/OFF control; exact original request,step and epoch ownership."""
from pathlib import Path
import base64,hashlib,json,math,os,signal,socket,subprocess,sys,time,types
wire=json.loads(base64.b64decode(WIRE,validate=True));claim=wire['claim'];label=claim['label'];parent=claim['parent'];node=claim['node']
assert (label,parent,node)==('i03-deep-offoff-repeatability-v1','1087656','f12r3n12') and claim['gpu_step_count_max']==1
blobs={n:base64.b64decode(v,validate=True) for n,v in wire['files'].items()}
assert set(blobs)==set(claim['files_SHA256']) and all(hashlib.sha256(v).hexdigest()==claim['files_SHA256'][n] for n,v in blobs.items())
COLUMNS=['step','label','state','exit_code','elapsed_seconds','node']
epoch=time.time();anchor=time.monotonic();deadline=epoch+7400;gpu_deadline=epoch+7200
ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
W=ROOT/(label+'-controller');assert not W.exists();W.mkdir(mode=0o700)
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),host=socket.gethostname())
def save(name,v):
 with (W/name).open('x') as f:json.dump(v,f,indent=2)
def compact(v):return json.dumps(v,separators=(',',':'),sort_keys=True).encode()
def emit(stage,**v):print(json.dumps(dict(stage=stage,request_id=REQUEST,**v)),flush=True)
def left(reserve=0):
 remaining=min(deadline-time.time(),7400-(time.monotonic()-anchor))-reserve
 if remaining<=0:raise TimeoutError('original controller epoch exhausted')
 return remaining
save('intent.json',dict(claim_commit=CLAIM,request_id=REQUEST,claim=claim,epoch=epoch,deadline=deadline,gpu_deadline=gpu_deadline,owner=owner,replay_allowed=False))
emit('REMOTE_INTENT_CREATED',owner=owner,epoch=epoch,deadline=deadline,gpu_deadline=gpu_deadline)
module=types.ModuleType('slurm_accounting');exec(compile(blobs['slurm_accounting.py'],'slurm_accounting.py','exec'),module.__dict__)
wait_terminal=module.wait_terminal
def check(argv,timeout=15):
 r=subprocess.run(argv,capture_output=True,timeout=min(timeout,left(2)))
 if r.returncode:raise RuntimeError('bounded '+argv[0]+' returned '+str(r.returncode))
 assert len(r.stdout)<=2*1024**2
 return r.stdout.decode()
def accounting(timeout=12):
 raw=check(['sacct','-n','-P','-j',parent,'--format=JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500'],timeout)
 rows=[]
 for line in raw.splitlines():
  fields=[part.strip() for part in line.split('|')]
  if len(fields)>=6 and fields[0].startswith(parent+'.'):rows.append(dict(zip(COLUMNS,fields[:6])))
 return rows
def terminal(name,maximum):
 end=time.monotonic()+min(maximum,left(5));seen=[]
 def read():
  remaining=end-time.monotonic()
  if remaining<=0:return None
  rows=[r for r in accounting(timeout=min(12,maximum,remaining)) if r['label']==name]
  if len(rows)>1:raise RuntimeError('duplicate original Slurm label')
  row=rows[0] if rows else None;seen.append(row)
  return [row[k] for k in COLUMNS] if row else None
 result=wait_terminal(read,end,interval=.5,max_observations=max(1,math.ceil(maximum*2)+1))
 emit('terminal_accounting',label=name,accounting_columns=COLUMNS,observations=seen,terminal_visible=result['terminal_visible'])
 return dict(zip(COLUMNS,result['row'])) if result['terminal_visible'] else None
def active():
 raw=check(['squeue','--steps','-h','-j',parent,'-w',node,'-o','%i|%j|%N'])
 for line in raw.splitlines():
  fields=line.split('|');assert len(fields)==3
  if fields[0] not in (parent+'.batch',parent+'.extern'):raise RuntimeError('active target-node step')
split_pin=claim['files_SHA256']['split_admission.py']
module_code=('import base64,hashlib,socket,sys,types;'
 "m=types.ModuleType('split_admission');sys.modules[m.__name__]=m;"
 'p=base64.b64decode('+repr(base64.b64encode(blobs['split_admission.py']).decode())+');'
 'assert hashlib.sha256(p).hexdigest()=='+repr(split_pin)+';'
 "exec(compile(p,'split_admission.py','exec'),m.__dict__);")
python='/usr/local/python3.12/bin/python3'
def snapshot(name):
 code="with socket.socket() as port:\n port.bind(('127.0.0.1',43888))\nimport json;print(json.dumps(m.snapshot(False)),flush=True)"
 raw=check(['srun','--jobid='+parent,'--job-name='+name,'--exact','--cpu-bind=none','--nodes=1','--ntasks=1','--cpus-per-task=2','--nodelist='+node,
  '--gres=none','--mem=8192M','--immediate=5','--time=00:01:00',python,'-B','-u','-c',module_code+'\n'+code],78)
 rows=[json.loads(v) for v in raw.splitlines() if v.startswith('{')];assert len(rows)==1;row=rows[0]
 acc=terminal(name,22)
 assert acc and acc['step']==row['step'] and acc['state']=='COMPLETED' and acc['exit_code']=='0:0' and acc['node']==node
 assert row['kfd_holder_count']==0 and len(row['pci'])==8
 return dict(row,accounting_state=acc['state'],exit_code=acc['exit_code'])
child=None;owned=None;gpu_acc=None;before=None;post=None;error=None;forced=False;stdout=b'';stderr=b''
try:
 row=check(['squeue','-h','-j',parent,'-o','%T|%D|%N']).strip().split('|');assert row[:2]==['RUNNING','40'] and len(row)==3
 nodes=check(['scontrol','show','hostnames',row[2]]).splitlines();assert len(set(nodes))==40 and node in nodes
 assert not ({label,label+'-pre',label+'-post'} & {r['label'] for r in accounting()})
 assert not (ROOT/(label+'-private')).exists();active();before=snapshot(label+'-pre');save('before.json',before);active()
 gate=('g=types.ModuleType("pilot_host_gate");p=base64.b64decode('+repr(base64.b64encode(blobs['pilot_host_gate.py']).decode())+');'
  'assert hashlib.sha256(p).hexdigest()=='+repr(claim['files_SHA256']['pilot_host_gate.py'])+';'
  "exec(compile(p,'pilot_host_gate.py','exec'),g.__dict__);raise SystemExit(g.main())")
 names={'alias_policy.py','alias_summary.py','diagnostic_binding.py','input_binding.json','pilot_inputs.py','pilot_observation.py','pilot_run.py','scoring.py','service_config.json','wire_client_r290.py'}
 service=compact({n:base64.b64encode(blobs[n]).decode() for n in sorted(names)});assert len(service)<=131072
 before_raw=compact(before);envelope=compact(dict(before=base64.b64encode(before_raw).decode(),inside=base64.b64encode(blobs['pilot_inside.sh']).decode(),probe=base64.b64encode(service).decode()))
 assert gpu_deadline-time.time()>7000
 options=list(claim['srun_options']);i=next(i for i,v in enumerate(options) if v.startswith('--time='));options[i]='--time='+str(max(1,math.ceil((gpu_deadline-time.time())/60)))
 argv=['srun',*options,python,'-B','-u','-c',module_code+gate,'--before-sha256',hashlib.sha256(before_raw).hexdigest(),
  '--inside-sha256',claim['files_SHA256']['pilot_inside.sh'],'--probe-sha256',hashlib.sha256(service).hexdigest(),'--deadline-epoch',str(gpu_deadline)]
 save('gpu-intent.json',dict(request_id=REQUEST,epoch=epoch,deadline=gpu_deadline,argv_SHA256=hashlib.sha256(compact(argv)).hexdigest(),before_SHA256=hashlib.sha256(before_raw).hexdigest(),claim_commit=CLAIM))
 emit('GPU_DISPATCH_INTENT',epoch=epoch,deadline=gpu_deadline,label=label)
 child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
 fields=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split();owned=dict(PID=child.pid,PGID=int(fields[2]),startticks=int(fields[19]));save('owned-srun.json',owned)
 try:stdout,stderr=child.communicate(input=envelope,timeout=min(gpu_deadline-time.time(),left(180)))
 except subprocess.TimeoutExpired as failure:
  forced=True;stdout=failure.stdout or b'';stderr=failure.stderr or b''
  for sig,budget in ((signal.SIGTERM,5),(signal.SIGKILL,5)):
   if child.poll() is not None:break
   current=Path('/proc',str(owned['PID']),'stat').read_text().split(') ',1)[1].split();assert int(current[19])==owned['startticks']
   try:os.killpg(owned['PGID'],sig)
   except ProcessLookupError:pass
   try:child.wait(timeout=min(budget,left(150)))
   except subprocess.TimeoutExpired:pass
  child.wait(timeout=min(2,left(145)))
 with (W/'gpu-stdout.private.log').open('xb') as f:f.write(stdout)
 with (W/'gpu-stderr.private.log').open('xb') as f:f.write(stderr)
 emit('GPU_SRUN_RETURN',returncode=child.returncode,forced_cleanup=forced)
 gpu_acc=terminal(label,22)
 if gpu_acc is None:
  original=[r for r in accounting() if r['label']==label]
  assert len(original)<=1
  if original:
   assert original[0]['node']==node and original[0]['step'].startswith(parent+'.')
   check(['scancel',original[0]['step']],10);forced=True;gpu_acc=terminal(label,22)
 if gpu_acc is None:raise RuntimeError('original GPU accounting remains UNKNOWN')
 post=snapshot(label+'-post');assert post['pci']==before['pci'] and post['boot_id']==before['boot_id'];save('post.json',post)
except BaseException as failure:error=dict(type=type(failure).__name__)
finally:
 if child is not None and child.poll() is None and owned is not None:
  forced=True
  for sig,budget in ((signal.SIGTERM,5),(signal.SIGKILL,5)):
   if child.poll() is not None:break
   try:
    current=Path('/proc',str(owned['PID']),'stat').read_text().split(') ',1)[1].split();assert int(current[19])==owned['startticks']
    os.killpg(owned['PGID'],sig)
   except (FileNotFoundError,ProcessLookupError):pass
   try:child.wait(timeout=min(budget,left(20)))
   except subprocess.TimeoutExpired:pass
out=dict(claim_commit=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,gpu_deadline=gpu_deadline,owner=owner,owned_srun=owned,
 actual_GPU_srun_launched=child is not None,returncode=child.returncode if child else None,forced_cleanup=forced,error=error,
 accounting_columns=COLUMNS,original_GPU_accounting=gpu_acc,before=before,post=post,wall_seconds=time.monotonic()-anchor,
 original_stdout_SHA256=hashlib.sha256(stdout).hexdigest(),original_stderr_SHA256=hashlib.sha256(stderr).hexdigest(),
 Model_GO=False,performance_qualified=False,quality_qualified=False)
out['passed']=bool(child and child.returncode==0 and not forced and error is None and gpu_acc and gpu_acc['state']=='COMPLETED' and gpu_acc['exit_code']=='0:0' and post)
save('outcome.json',out);emit('CONTROLLER_TERMINAL',result=out)
raise SystemExit(0 if out['passed'] else 1)
