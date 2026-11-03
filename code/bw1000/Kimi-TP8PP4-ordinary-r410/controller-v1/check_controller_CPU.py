"""Actual one-target Controller/collector and Popen argv; recording Slurm only."""
from pathlib import Path
import ast,ctypes,json,os,signal,subprocess,tempfile,time
from unittest.mock import patch
from contract_r410 import *
from controller import Controller
from slurm_backend import SlurmBackend,clean_env
D=Path(__file__).resolve().parent
def plan():
 p=json.loads((D/'PLAN.template.json').read_bytes());p.update(GPU_GO=True,Model_GO=True,execute_once=True,publication_commit='a'*40,controller_source_manifest_SHA256='b'*64);return p
class Clock:
 def __init__(self):self.now=1000.
 def __call__(self):return self.now
 def sleep(self,n):self.now+=n
class Proc:
 pid=888888
 def __init__(self,code=None):self.code=code
 def poll(self):return self.code
 def wait(self,timeout=None):self.code=0;return 0
class Backend:
 def __init__(self,root,mode):self.root=root;self.mode=mode;self.started=0;self.cancelled=0;self.posts=[];self.observed=0;self.actual='910380.12345';self.proc=Proc()
 def final_preflight(self):return dict(passed=True,recording_CPU_contract=True,Slurm_commands=0)
 def validate_resources(self):pass
 def no_foreign(self):pass
 def idle(self,phase):
  self.posts.append(phase);return[dict(node=n,ib_ipv4=['12.1.14.'+str(i)],actual_step='910380.12344')for i,n in enumerate(TARGET_NODES)]
 def start(self,env,epoch):
  self.started+=1;assert env['KIMI410_NATIVE_DEADLINE_UNIX']==str(epoch+5100)and env['KIMI410_SHELL_DEADLINE_UNIX']==str(epoch+5280)
  assert env['WAVE_INSIDE_R2323'].endswith('controller-v1/GPU_inside.sh');return dict(label='recording',srun_client_PID=self.proc.pid)
 def observe(self):
  self.observed+=1
  if self.mode=='transient'and self.observed==1:raise subprocess.TimeoutExpired(['sacct'],1)
  if self.mode=='terminal_before_exit'and self.observed==1:return dict(row=[self.actual,'label','COMPLETED','0:0',','.join(TARGET_NODES)],srun_exit_code=None)
  self.proc.code=1 if self.mode=='step_failed'else 0
  return dict(row=[self.actual,'label','FAILED'if self.mode=='step_failed'else'COMPLETED','1:0'if self.mode=='step_failed'else'0:0',','.join(TARGET_NODES)],srun_exit_code=self.proc.code)
 def cancel_owned(self):self.cancelled+=1;return[dict(recording=True)]
def outputs(root,run):
 from collect import sha
 historical=json.loads((D.parent/'runtime-v1/CPU_recorded_outputs.json').read_bytes())['outputs'];requests=[]
 for i,name in enumerate(['retained8635','near32k']):
  ids=historical[name];requests.append(dict(name=name,input_tokens=[8635,32638][i],output_budget=128,completed=True,normal_reset=True,finish='eos',quality=dict(passed=True),generated_ids=ids,prompt_ID_SHA256=PROMPT_SHA[i],output_ID_SHA256=__import__('hashlib').sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),events=[dict(at_s=j*.1,token=v)for j,v in enumerate(ids)],full_request_s=10.))
 count=197
 for node in range(4):
  q=dict(completed=True,all8_owned_workers_shutdown_completed=True,sampler_CPU_cleanup_completed=True,control_CPU_cleanup_completed=True,cleanup_errors=[],actual_step='910380.12345',node=node,source_manifest_SHA256=RUNTIME_SHA,Native_SHA256=NATIVE_SHA,layout=dict(TP=8,PP=4,PP_stage=node,sampling_node=3),all93_stage_keys=dict(all93_layer_ids_exactly_once_across_PPstages=True),owner_affinity=dict(verified=True),last_memory_guard=dict(minimum_free_bytes=5<<30),ordinary_report=dict(completed=True,draft_cards=0,requests=requests),ordinary_calls=[dict(serial=i,validated=True,CAPI_return=0)for i in range(count)],output_metrics=dict(engine_calls=count))
  (root/'results'/('Kimi-ordinary-r410-'+run+'-node'+str(node)+'.json')).write_text(json.dumps(q))
def case(mode):
 with tempfile.TemporaryDirectory(prefix='Kimi410-controller-')as tmp:
  root=Path(tmp);(root/'results').mkdir();(root/'control').mkdir();p=plan();clock=Clock();b=Backend(root,mode);outputs(root,p['run'])
  c=Controller(root,p,b,'0'*40,clock=clock,sleep=clock.sleep)
  if mode=='artifact_timeout':
   with patch('collect.collect',side_effect=TimeoutError('recording missing final artifact')):q=c.run()
  else:q=c.run()
  assert b.started==1 and b.posts[0]=='before'
  if mode in['step_failed','artifact_timeout']:assert not q['passed']and b.cancelled==1 and any(v in b.posts for v in['post','failure-post'])
  else:assert q['passed']and b.cancelled==0 and b.posts==['before','post']
  if mode=='transient':assert len(q['transient_observation_errors'])==1
  if mode=='terminal_before_exit':assert b.observed==2
  try:Controller(root,p,b,'0'*40,clock=clock,sleep=clock.sleep).run();raise AssertionError('duplicate accepted')
  except AssertionError:assert b.started==1
  return dict(case=mode,passed=True,start_count=b.started,cancel_count=b.cancelled,post_phases=b.posts)
def main():
 rows=[case(v)for v in['normal','transient','terminal_before_exit','step_failed','artifact_timeout']]
 # Real start implementation: correct parent4node/32GRES/64CPU/128GB plus absolute remaining wall.
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);(root/'logs').mkdir();p=plan();b=SlurmBackend(root,D,p);captured=[]
  with patch('slurm_backend.time.time',return_value=1002),patch('slurm_backend.subprocess.Popen',side_effect=lambda a,**kw:(captured.append((a,kw))or Proc())):
   b.start(dict(KIMI410_ORDINARY_RUNTIME='1'),1000)
  argv,kw=captured[0];assert'--jobid=910380'in argv and'-N4'in argv and'-n4'in argv and'--gres=dcu:8'in argv and'-c64'in argv and'--mem=128000M'in argv and argv[argv.index('-t')+1]=='01:29:58'and kw['start_new_session']is True
  b.account=lambda:[b.actual or'910380.12345','label','RUNNING','0:0',','.join(TARGET_NODES)];kills=[];commands=[]
  with patch.object(signal,'SIGKILL',9,create=True),patch('slurm_backend.subprocess.run',side_effect=lambda a,**kw:commands.append(a)),patch('slurm_backend.os.killpg',side_effect=lambda pid,sig:kills.append((pid,sig)),create=True):b.cancel_owned()
  assert commands==[['scancel','--signal=TERM','910380.12345']]and kills==[(Proc.pid,signal.SIGTERM),(Proc.pid,0),(Proc.pid,9)]
  rows.append(dict(case='actual_start_argv_and_original_group_after_leader_exit',passed=True))
 with patch.dict(os.environ,dict(BASH_ENV='recording-hook',APPTAINERENV_BASH_ENV='recording-nested-hook',KIMI410_PREFLIGHT_CPU_ONLY='1'),clear=False):
  env=clean_env();assert'BASH_ENV'not in env and'APPTAINERENV_BASH_ENV'not in env and'KIMI410_PREFLIGHT_CPU_ONLY'not in env
 tree=ast.parse((D.parent/'runtime-v1/bootstrap.py').read_bytes());mainfn=next(v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='main');defaults=dict(zip([v.arg for v in mainfn.args.args][-len(mainfn.args.defaults):],mainfn.args.defaults));assert isinstance(defaults['providers'],ast.Constant)and defaults['providers'].value is None
 last=tree.body[-1];call=last.body[0].value;assert isinstance(call,ast.Call)and call.func.id=='main'and not call.args and not call.keywords
 rows.append(dict(case='production_bootstrap_defaultprovidersNone_no_preflight_hook',passed=True))
 print(json.dumps(dict(completed=True,passed=True,cases=rows,Slurm_commands=0,GPU=0,Native_dlopen=False,Runtime_created=False,scope='Actual Controller/Popen/collector with recording Slurm and synthetic final report leaf; no device/model qualification')))
if __name__=='__main__':main()
