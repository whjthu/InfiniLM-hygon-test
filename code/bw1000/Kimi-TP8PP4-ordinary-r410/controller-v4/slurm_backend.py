"""Existing347 owned-process backend, only one32target ordinary residency."""
from pathlib import Path
import json,math,os,re,signal,subprocess,time
from contract_r410 import TARGET_NODES,CPU_NODE,PARENT,CPU_PARENT,PORTS,GPU_SECONDS,TERMINAL,sha
def clean_env():
 out={}
 for k,v in os.environ.items():
  key=k
  for prefix in['APPTAINERENV_','SINGULARITYENV_']:
   if key.startswith(prefix):key=key[len(prefix):];break
  if key in['BASH_ENV','ENV']or key.startswith(('KIMI410_','BASH_FUNC_','CUDA_','HIP_','ROCR_','GPU_')):continue
  if k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in['SLURM_CONF','SLURM_CONF_SERVER']:continue
  out[k]=v
 return out
class SlurmBackend:
 def __init__(self,root,directory,plan):
  self.root=Path(root);self.directory=Path(directory);self.plan=plan;self.env=clean_env();self.proc=None;self.actual=None;self.ignored=set();self.forced=[];self.probe_records=[]
 def command(self,args,timeout=25):return subprocess.check_output(args,env=self.env,text=True,timeout=timeout)
 def expand(self,nodes):return self.command(['scontrol','show','hostnames',nodes],15).splitlines()
 def label(self,suffix='target'):return'Kimi-ordinary-r410-'+self.plan['run']+'-'+suffix
 def rows(self):
  raw=self.command(['sacct','-S','2026-10-31','-j',str(PARENT),'-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,NodeList%120'])
  return[s.split('|')for s in raw.splitlines()if s.strip()]
 def account(self):
  rows=[r for r in self.rows()if r[1]==self.label()];assert len(rows)<=1,'Unknown original submission is not restart authority'
  if not rows:return None
  r=rows[0];assert re.fullmatch(str(PARENT)+r'[.]\d+',r[0])and set(self.expand(r[4]))==set(TARGET_NODES)
  assert self.actual in[None,r[0]];self.actual=r[0];return r
 def no_foreign(self):
  for r in self.rows():
   if not re.fullmatch(str(PARENT)+r'[.]\d+',r[0])or r[0]in self.ignored or r[0]==self.actual or r[2].split()[0].rstrip('+')in TERMINAL:continue
   assert not set(self.expand(r[4]))&set(TARGET_NODES),'Active foreign step on claimed node'
 def parent(self,number,nodes):
  raw=self.command(['scontrol','show','job',str(number),'-o']);assert'UserId=aclsylqidf('in raw and'JobState=RUNNING'in raw
  found=re.search(r'(?:^| )NodeList=([^ ]+)',raw);assert found and set(nodes)<=set(self.expand(found.group(1)))
  return raw
 def validate_resources(self):
  self.parent(PARENT,TARGET_NODES);self.parent(CPU_PARENT,[CPU_NODE]);assert self.account()is None;self.no_foreign()
 def final_preflight(self):
  from contract_r410 import publish_once
  prefix='Kimi-ordinary-r410-'+self.plan['run'];planfile=self.root/'results'/(prefix+'-preflight-plan.json');result=self.root/'results'/(prefix+'-final-CLI-preflight.json')
  assert not result.exists();publish_once(planfile,self.plan)
  env=dict(self.env,WAVE_INSIDE_R2323='Kimi-TP8PP4-ordinary-r410/controller-v4/CPU_preflight_inside.sh',KIMI410_PREFLIGHT_PLAN='/refactor/results/'+planfile.name,KIMI410_PREFLIGHT_OUTPUT='/refactor/results/'+result.name,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
  for key in['SLURM_JOB_ID','SLURM_STEP_ID','SLURM_PROCID']:env[key]=os.environ[key]
  assert env['SLURM_JOB_ID']=='910407'
  # Plain owned CPU keeper/container subprocess; this CLI issues no Slurm command.
  p=subprocess.run(['bash',str(self.directory/'outer.sh'),'probe'],env=env,text=True,capture_output=True,timeout=240)
  log=self.root/'logs'/(prefix+'-final-CLI-preflight.log');log.write_text(p.stdout+p.stderr)
  assert p.returncode==0 and result.is_file(),dict(rc=p.returncode,log=str(log))
  q=json.loads(result.read_bytes());assert q['passed']and q['Slurm_commands']==0 and q['GPU']==0 and q['Model_GO']is False and not q['Native_dlopen']and q['controller_source_manifest_SHA256']==self.plan['controller_source_manifest_SHA256']
  return dict(path=str(result),SHA256=sha(result),log=str(log),log_SHA256=sha(log),passed=True,Slurm_commands=0,GPU=0,Native_dlopen=False,nodes=4,CPU826_not_repeated=True)
 def idle(self,phase):
  cmd=['srun','--jobid='+str(PARENT),'--overlap','--exact','--immediate=30','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(TARGET_NODES),'-t','00:02:00','--job-name='+self.label(phase),'python3','-B',str(self.directory/'resource_probe.py'),'--phase',phase]
  out=self.root/'logs'/(self.label(phase)+'-probe.stdout');err=self.root/'logs'/(self.label(phase)+'-probe.stderr')
  assert not out.exists()and not err.exists()
  try:p=subprocess.run(cmd,env=self.env,text=True,capture_output=True,timeout=115)
  except subprocess.TimeoutExpired as e:
   out.write_bytes(e.stdout if isinstance(e.stdout,bytes)else(e.stdout or '').encode());err.write_bytes(e.stderr if isinstance(e.stderr,bytes)else(e.stderr or '').encode());self.probe_records.append(dict(phase=phase,stdout=str(out),stdout_SHA256=sha(out),stderr=str(err),stderr_SHA256=sha(err),timeout=True));raise
  out.write_text(p.stdout);err.write_text(p.stderr);self.probe_records.append(dict(phase=phase,stdout=str(out),stdout_SHA256=sha(out),stderr=str(err),stderr_SHA256=sha(err),returncode=p.returncode))
  rows=[json.loads(s)for s in p.stdout.splitlines()if s.startswith('{')]
  assert p.returncode==0 and len(rows)==4 and {r['node']for r in rows}==set(TARGET_NODES)and len({r['actual_step']for r in rows})==1
  self.ignored.update(r['actual_step']for r in rows)
  assert all(r['CPU_only']and not r['GPU_initialized']and r['passed']and r['KFD_empty']and not r['matching_bootstrap_PIDs']for r in rows)
  assert len({(r['node'],v['PCI'])for r in rows for v in r['devices']})==32
  return rows
 def start(self,values,epoch):
  assert self.proc is None,'One original residency only';remaining=math.floor(epoch+GPU_SECONDS-time.time());assert remaining>180
  wall='%02d:%02d:%02d'%(remaining//3600,(remaining%3600)//60,remaining%60)
  env=dict(self.env,**values)
  cmd=['srun','--jobid='+str(PARENT),'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c64','-w',','.join(TARGET_NODES),'--gres=dcu:8','--mem=128000M','-t',wall,'--job-name='+self.label(),'--output='+str(self.root/'logs'/(self.label()+'-%J-%t.log')),'--export=ALL','bash',str(self.directory/'outer.sh'),'probe']
  with(self.root/'logs'/(self.label()+'-srun-client.log')).open('x')as log:self.proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  return dict(srun_client_PID=self.proc.pid,original_owned_PGID=self.proc.pid,argv=cmd,label=self.label())
 def observe(self):return dict(row=self.account(),srun_exit_code=self.proc.poll())
 def cancel_owned(self):
  try:
   r=self.account()
   if r and r[2].split()[0].rstrip('+')not in TERMINAL:
    subprocess.run(['scancel','--signal=TERM',r[0]],env=self.env,check=True,timeout=25);self.forced.append(dict(actual_step=r[0],signal='TERM'))
  except BaseException as e:self.forced.append(dict(account_cleanup_error=repr(e)))
  if self.proc is not None:
   for sig in[signal.SIGTERM,signal.SIGKILL]:
    try:os.killpg(self.proc.pid,sig)
    except ProcessLookupError:break
    self.forced.append(dict(original_owned_PGID=self.proc.pid,signal=int(sig)))
    try:self.proc.wait(timeout=20 if sig==signal.SIGTERM else 10)
    except subprocess.TimeoutExpired:continue
    if sig==signal.SIGTERM:
     try:os.killpg(self.proc.pid,0)
     except ProcessLookupError:break
     continue
    break
  return self.forced
