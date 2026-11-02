"""Actual r405 controller with recording Slurm/Popen, real temp artifacts; no GPU/Runtime."""
from pathlib import Path
import importlib.util,json,tempfile,types,subprocess

def main():
 spec=importlib.util.spec_from_file_location('actual405controller',Path(__file__).with_name('controller.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 calls=dict(account=0,starts=0,scancel=0,probes=0)
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);(root/'results').mkdir();(root/'logs').mkdir();step='910371.999999'
  worker=dict(completed=True,normal_close=True,PID=10101,samples=[dict(passed=True,call_completed=True,counter_passed=True,numerical=dict(passed=True))]*320,projection_counts=dict(chosen=160,fallback=0,consumer_counts=[160]*4))
  for node in range(2):(root/'results'/('Kimi405-C1-Delta-projection-'+step+'-node'+str(node)+'.json')).write_text(json.dumps(dict(completed=True,normal_close=True,owners=8,Math=0,capture=False,workers=[worker]*8)))
  c.verify=lambda r,g:(dict(CPU_receipt_SHA256='c'*64,library_SHA256='d'*64,CPU_source_manifest_SHA256='e'*64),{})
  original_start=c.start
  def start(r,pin):
   calls['starts']+=1
   def popen(args,**kw):
    assert '-N2'in args and '-c32'in args and '--gres=dcu:8'in args
    assert kw['start_new_session'] and kw['env']['KIMI_ENGINE_HEAD']=='12.1.2.3' and kw['env']['KIMI405_GPU_MANIFEST_SHA']==pin
    assert kw['env']['WAVE_INSIDE_R2323']==c.NAME+'/GPU_inside.sh'
    return types.SimpleNamespace(pid=424242,poll=lambda:0)
   return original_start(r,pin,popen=popen)
  c.start=start;c.time.sleep=lambda _:None
  def command(args,**kw):
   if args[:4]==['scontrol','show','job','910371']:return 'UserId=aclsylqidf(1000) JobState=RUNNING NodeList=a02r3n[03,07]'
   if args[:3]==['scontrol','show','hostnames']:return 'a02r3n03\na02r3n07\n'
   if args[0]=='sacct':
    calls['account']+=1
    if calls['account']<=2:return ''
    if calls['account']==3:raise subprocess.TimeoutExpired(args,30)
    return step+'|'+c.LABEL+'|COMPLETED|0:0|a02r3n[03,07]\n'
   raise AssertionError(args)
  def run(args,**kw):
   if args[0]=='scancel':calls['scancel']+=1;raise AssertionError('Healthy original step must not be cancelled')
   assert args[0]=='srun'and '--gres=none'in args and '-N2'in args;calls['probes']+=1
   return types.SimpleNamespace(returncode=0,stderr='',stdout='\n'.join(json.dumps(dict(node=n,passed=True,ib0_IPv4='12.1.2.3'if i==0 else'12.1.2.4',KFD_pids=[]))for i,n in enumerate(c.NODES)))
  c.subprocess.check_output=command;c.subprocess.run=run
  q=c.run(root,dict(execute_once=True,publication_commit='f'*40,GPU_source_manifest_SHA256='a'*64))
  assert q['passed'] and q['normal16'] and len(q['owners'])==2 and len(q['submissions'])==1 and len(q['observation_errors'])==1
  assert calls==dict(account=4,starts=1,scancel=0,probes=2),calls
 return dict(completed=True,passed=True,actual_Controller_run=True,actual_start_Popen_argv_env=True,transient_account_same_handle=True,normal16_recording=True,start1=True,cancel0=True,probe2=True,Runtime=False,GPU=False,Slurm=False,qualification='recording lifecycle only, real temp files; no Native/collective/device proof')
if __name__=='__main__':print(json.dumps(main()))
