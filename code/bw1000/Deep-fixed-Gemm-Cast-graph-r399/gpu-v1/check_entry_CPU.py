"""Actual Python session and Controller.run, recording CAPI/Slurm only; no new CPU job."""
from pathlib import Path
import importlib.util,json,tempfile,threading,types,time
D=Path(__file__).resolve().parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 d=load('actual_r399_gpu_glue',D/'GPU_driver.py');checks=[]
 class Fake:
  def __init__(self,fail=0):self.owner=threading.get_ident();self.c={'prepares':0,'captures':0,'eager_calls':0,'replay_calls':0,'phase':2};self.closed=0;self.calls=0;self.fail=fail
  def inputs(self,*v):
   if threading.get_ident()!=self.owner:raise RuntimeError('recording owner refusal')
  def prepare(self):self.c['prepares']+=1
  def capture(self):self.c['captures']+=1
  def run(self,m,n):
   self.calls+=1
   if self.fail and self.calls==self.fail:raise RuntimeError('recording CAPI execution failure')
   self.c['replay_calls'if m else'eager_calls']+=n;return .1
  def read(self):return'recording-output'
  def check(self):pass
  def report(self):return dict(self.c)
  def close(self):self.closed+=1
 b=Fake();out=d.execute_session(b,[(1,2),(3,4),(0,4)],lambda i,v:{'passed':True},lambda v:None,time.time()+60);assert out['normal_runtime_close']and b.closed==1 and b.c['eager_calls']==99 and b.c['replay_calls']==100;checks.append('actual_execute_session_3patterns_1prepare_1capture_99eager_100replay_ownerrefusal_close1')
 for kind in ['execution','numeric']:
  b=Fake(3 if kind=='execution'else 0)
  try:d.execute_session(b,[(1,2)]*3,lambda i,v:{'passed':kind!='numeric'},lambda v:None,time.time()+60)
  except(RuntimeError,AssertionError):pass
  else:raise AssertionError('failure did not stop')
  assert b.closed==1 and b.calls==(3 if kind=='execution'else 1);checks.append(kind+'_failure_stop_no_retry_close1')
 c=load('actual_r399_controller',D/'controller.py');calls={'account':0,'starts':0,'scancel':0,'probes':0};proc=types.SimpleNamespace(pid=424242,poll=lambda:0)
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);(root/'results').mkdir();(root/'logs').mkdir();step='910374.999999';a=dict(completed=True,passed=True,PID=10001,normal_runtime_close=True,Torch_after_Runtime_close=True,GPU_count=1,Model_loaded=False,session={'counters':{'prepares':1,'captures':1,'eager_calls':99,'replay_calls':100},'foreign_thread_refused_without_owner_poison':True});(root/'results'/('Deep399-'+step+'.json')).write_text(json.dumps(a))
  c.verify=lambda r,g:({'CPU_receipt_SHA256':'c'*64,'library_SHA256':'d'*64},{})
  def start(r,pin):calls['starts']+=1;return proc,['recording-srun']
  c.start=start;c.time.sleep=lambda s:None
  def check_output(args,**kwargs):
   if args[:4]==['scontrol','show','job','910374']:return'UserId=aclsylqidf(1000) JobState=RUNNING NodeList=a02r4n07'
   if args[:3]==['scontrol','show','hostnames']:return'a02r4n07\n'
   if args[0]=='sacct':
    calls['account']+=1
    if calls['account']<=2:return''
    if calls['account']==3:raise c.subprocess.TimeoutExpired(args,30)
    return step+'|'+c.LABEL+'|COMPLETED|0:0|a02r4n07\n'
   raise AssertionError(args)
  def run(args,**kwargs):
   if args[0]=='scancel':calls['scancel']+=1;raise AssertionError('healthy original step canceled')
   assert args[0]=='srun'and'--gres=none'in args;calls['probes']+=1;return types.SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(dict(node='a02r4n07',passed=True,KFD_pids=[])))
  c.subprocess.check_output=check_output;c.subprocess.run=run
  q=c.run(root,dict(execute_once=True,publication_commit='f'*40,GPU_source_manifest_SHA256='a'*64));assert q['passed']and q['normal1']and len(q['submissions'])==1 and q['actual_step']==step and len(q['observation_errors'])==1 and calls==dict(account=4,starts=1,scancel=0,probes=2);checks.append('actual_Controller_transientaccount_restore_originalhandle_start1_cancel0_normalpost')
 print(json.dumps(dict(completed=True,passed=True,cases=checks,Slurm=False,GPU=False,Runtime=False,CAPI_and_Slurm_leaves='recording only',actual_Python_entry_and_controller=True)))
if __name__=='__main__':main()
