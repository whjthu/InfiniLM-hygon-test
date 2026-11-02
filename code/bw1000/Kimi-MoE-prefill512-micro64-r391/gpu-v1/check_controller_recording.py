"""One actual Controller.run scenario; Slurm/Popen/owner files are recording leaves."""
from pathlib import Path
import importlib.util,json,tempfile,types
D=Path(__file__).resolve().parent
def main():
 spec=importlib.util.spec_from_file_location('r391_controller_transient',D/'controller.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);calls={'account':0,'starts':0,'scancel':0,'probes':0};proc=types.SimpleNamespace(pid=424242,poll=lambda:0)
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);(root/'results').mkdir();(root/'logs').mkdir();step='910371.999999'
  for node in range(2):
   trials=[dict(counters_pass=True,input_readonly=True,output_tail_canary=True,error_vs_A={'screen_pass':True},ownrepeat={'screen_pass':True})for _ in range(102)]
   ranks=[dict(global_rank=node*8+r,completed=True,normal_Runtime_close=True,module_and_observer_closed=True,trials=trials)for r in range(8)]
   a=dict(completed=True,node=node,PID=10000+node,normal_CUDA_cleanup=True,Torch_after_Runtime_close=True,native=dict(completed=True,all8normal_before_communicators=True,reports=ranks))
   (root/'results'/('Kimi391-'+step+'-node%d.json'%node)).write_text(json.dumps(a))
  c.verify=lambda r,g:({'CPU_receipt_SHA256':'c'*64,'library_SHA256':'d'*64},{})
  def start(r,pin):calls['starts']+=1;return proc,['recording-srun']
  c.start=start;c.time.sleep=lambda seconds:None
  def check_output(args,**kwargs):
   if args[:4]==['scontrol','show','job','910371']:return'UserId=aclsylqidf(1000) JobState=RUNNING NodeList=a02r3n03,a02r3n07'
   if args[:3]==['scontrol','show','hostnames']:return'a02r3n03\na02r3n07\n'
   if args[0]=='sacct':
    calls['account']+=1
    if calls['account']<=2:return''
    if calls['account']==3:raise c.subprocess.TimeoutExpired(args,30)
    return step+'|'+c.LABEL+'|COMPLETED|0:0|a02r3n03,a02r3n07\n'
   raise AssertionError(args)
  def run(args,**kwargs):
   if args[0]=='scancel':calls['scancel']+=1;raise AssertionError('healthy original job canceled')
   assert args[0]=='srun'and'--gres=none'in args;calls['probes']+=1
   return types.SimpleNamespace(returncode=0,stderr='',stdout='\n'.join(json.dumps(dict(node=n,passed=True,ib0_IPv4='12.1.2.23',KFD_pids=[]))for n in c.NODES))
  c.subprocess.check_output=check_output;c.subprocess.run=run
  q=c.run(root,dict(execute_once=True,publication_commit='f'*40,GPU_source_manifest_SHA256='a'*64))
  assert q['passed']and q['normal16']and len(q['submissions'])==1 and q['actual_step']==step and len(q['observation_errors'])==1
  assert calls==dict(account=4,starts=1,scancel=0,probes=2)
 print(json.dumps(dict(completed=True,passed=True,actual_Controller_run=True,scenario='transientaccountTimeout_then_recover_same_handle_normal',calls=calls,recording_Slurm_Popen_and_owner_leaves=True,new_Slurm=False,GPU=False,Runtime=False,Native_dlopen=False,original_epoch_not_reset=True,source_not_restarted=True)))
if __name__=='__main__':main()
