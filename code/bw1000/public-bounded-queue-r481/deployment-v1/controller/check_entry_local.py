"""Actual controller.main/Run/JSON/commands, explicit Slurm/Native leaves."""
from pathlib import Path
import argparse,contextlib,hashlib,importlib.util,json,os,sys,tempfile
from types import SimpleNamespace
from unittest.mock import patch
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D));import controller as C,support
G=D.parents[4]
PROTOCOL=G/'code/bw1000/Deep-fixed-fullpre-experimental-profile-r454/source-v1/public_deep_model/protocol.py'
real_read=Path.read_text

def case(mode):
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);R=root/'remote';(R/'logs').mkdir(parents=True);(R/'results').mkdir();package=root/'package';(package/'public_deep_model').mkdir(parents=True);(package/'public_deep_model/__init__.py').write_text('');(package/'public_deep_model/protocol.py').write_bytes(PROTOCOL.read_bytes());sys.path.insert(0,str(package))
  node='a02r4n01';v=dict(run='A-OFF-recording',node=node,pair='A',queue_policy=False);cp=D/'configs/A-OFF.json';c=json.loads(cp.read_bytes());runtime=R/'results/Deep-bounded-queue-r481'/v['run'];label=C.LABEL+'-'+v['run'];base=R/'results'/label;out=base.with_name(base.name+'-client');m=dict(GPU_GO=True,Model_GO=True,private_plan=dict(path=str(root/'plan.json'),SHA256='recording-input'),files_SHA256={'finite_client.py':'recording-source'});state=dict(gpu=None,client=None,account_missing=False,cleanup=[],probes=[],stop_leaf=False)
  def put(p,q):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(q)+chr(10))
  class Process:
   def __init__(self,args,**kwargs):
    self.args=args;self.pid=99001 if '--gres=dcu:8'in args else 99002
    if self.pid==99001:
     state['gpu']=self;epoch=float(kwargs['env']['DEEP_R481_GPUSTART']);state['identity']=C.identity(v,'910374.9900',epoch,cp);runtime.mkdir(parents=True);put(runtime/'FRONTEND.json',dict(identity=state['identity'],PID=88001));put(runtime/'MODEL_READY.json',dict(identity=state['identity'],Model_READY=True,Native_internal_owner_count=8,Native_SHA256=C.NATIVE_SHA,HC_mode=0,fixed_profile=dict(fixed_mode=0,r366_fixed=0,HC_groups=86),queue_policy=dict(enabled=False,physical_pending_capacity=2,queue_wait_seconds=5.)))
    else:
     state['client']=self;config=json.loads(Path(args[args.index('--config')+1]).read_bytes());server=json.loads(Path(config['server_config']).read_bytes());assert server['runtime_directory']==str(runtime.parent)and server['provider_config_SHA256']==C.SHA(cp);assert all(server[k]==x for k,x in state['identity'].items());put(Path(args[args.index('--identity')+1]),dict(PID=self.pid));out.mkdir();put(out/'result.json',dict(completed=True,planned=20,warm=0,rows=[dict(index=i,score=dict(correct=False))for i in range(20)],quality_summary=dict(planned=20,correct=0)))
     if mode!='client_failure':
      from public_deep_model.protocol import Run
      Run(server).stop();assert json.loads((runtime/'STOP.json').read_bytes())['identity']==state['identity'];state['stop_leaf']=True
   def poll(self):return 1 if self.pid==99002 and mode=='client_failure'else 0 if self.pid==99002 or(runtime/'STOP.json').exists()else None
  def output(args,**kwargs):
   if args[:3]==['scontrol','show','job']:return 'JobState=RUNNING UserId=aclsylqidf(154459) NodeList='+node
   if args[:3]==['scontrol','show','hostnames']:return node+chr(10)
   if args[0]=='sacct':
    if 'JobIDRaw,ElapsedRaw,State,ExitCode'in args:return '910374.9900|12|COMPLETED|0:0'+chr(10)
    if state['gpu']is None:return ''
    if mode=='delayed_account'and state['client']is not None and not state['account_missing']:state['account_missing']=True;return ''
    s='COMPLETED'if state['gpu'].poll()==0 else'RUNNING';values=['910374.9900|'+label+'|'+s+'|0:0|'+node]
    if state['client']:values.append('910374.9901|'+label+'-client|'+('FAILED|1:0'if mode=='client_failure'else'COMPLETED|0:0')+'|'+node)
    return chr(10).join(values)+chr(10)
   raise AssertionError('Unrecorded command '+repr(args))
  def run(args,**kwargs):
   assert args[0]=='srun'and '--gres=none'in args;phase=args[args.index('--job-name='+label+'-before')]if '--job-name='+label+'-before'in args else'post';state['probes'].append(phase);return SimpleNamespace(returncode=0,stdout=json.dumps(dict(node=node,KFD_available=True,GPU_initialized=False,kfd_pids=[],matching_component_processes=[],expected_PIDs_still_exist=[],ports_bindable={'43882':True},physical_PCI=['0000:%02x:00.0'%i for i in range(8)]))+chr(10),stderr='')
  def read_text(p,*a,**k):return ' '.join(['0']*21+['12345'])if str(p).replace(chr(92),'/').startswith('/proc/')else real_read(p,*a,**k)
  def collect(runtime0,identity,*args,**kwargs):
   assert runtime0==runtime and identity==state['identity'];assert json.loads((runtime/'STOP.json').read_bytes())['identity']==identity;return dict(expected_PIDs=[88001,99002],recording_terminal_Native_evidence_leaf=True,quality_summary=dict(planned=20,correct=0),execution_passed=True)
  def close_owned(name,proc,step,exact,limit):state['cleanup'].append((name,proc.pid,step));return dict(recording_owned_cleanup_leaf=True,closed=True)
  import collect as terminal
  with patch.object(C,'R',R),patch.object(support,'R',R),patch.object(C,'PACKAGE',package),patch.object(C,'verify',lambda *a:(m,v,c,cp)),patch.object(C.socket,'gethostname',lambda:'a14r2n20'),patch.dict(os.environ,{'SLURM_JOB_ID':'910407','SLURM_STEP_ID':'9000'}),patch.object(C.subprocess,'check_output',output),patch.object(C.subprocess,'Popen',Process),patch.object(C.subprocess,'run',run),patch.object(C.time,'sleep',lambda *a:None),patch.object(Path,'read_text',read_text),patch.object(terminal,'collect',collect),patch.object(C,'close_owned',close_owned):
   rc=C.main(['--manifest-sha','a'*64,'--publication-commit','b'*40,'--instance','A-OFF'])
  watch=json.loads(base.with_name(base.name+'-watch.json').read_bytes());assert watch['completed']and watch['actual_GPU']=='910374.9900'and(runtime/'STOP.json').exists();assert watch['HTTP_planned']==20 and watch['warm']==0
  if mode=='client_failure':assert rc==1 and not watch['passed']and watch['forced_cleanup']and len(state['cleanup'])==2 and watch['STOP_owner']=='controller_failure_or_terminal_fallback'
  else:assert rc==0 and watch['passed']and not watch['forced_cleanup']and watch['client_quality_summary']['correct']==0 and watch['STOP_owner']=='client_or_existing_exactidentity'
  if mode=='delayed_account':assert state['account_missing']
  return dict(mode=mode,passed=True,actual_controller_main=True,actual_Run_identity_and_HOST_alias=True,quality_wrong_not_cleanup_gate=mode!='client_failure',recording_slurm_steps=2,Native=False,HTTP_external=False,GPU=False,terminal_Native_aggregate_and_owned_signals_are_recording_leaves=True)

def main():return dict(passed=True,cases=[case(m)for m in ['normal_allwrong','delayed_account','client_failure']],scope='Newactualcontroller main andrealRun/JSON/hostalias/config/command glue; Slurmprocesses/Native terminalcollector/owned signals are explicit recording leaves; no actualoldModel/CPU suite repeated')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();q=main();Path(a.output).write_text(json.dumps(q,indent=2)+chr(10));print(json.dumps(q))
