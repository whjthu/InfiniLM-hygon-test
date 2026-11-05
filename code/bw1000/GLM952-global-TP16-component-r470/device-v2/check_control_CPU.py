"""Run actual controller entry against only recording Slurm/device leaves."""
from pathlib import Path
import tempfile,sys,os,json,runpy,types,unittest.mock as M
D=Path(__file__).resolve().parent

def case(corrupt=False,cancel=False):
 sys.path.insert(0,str(D));module=runpy.run_path(str(D/'controller.py'));fn=module['main'];g=fn.__globals__;real_time=g['time'].time;NODES=g['NODES'];gpu='910371.9991';calls=[];signals=[]
 with tempfile.TemporaryDirectory(prefix='glm470-control-')as temp:
  root=Path(temp);(root/'logs').mkdir();(root/'results').mkdir();out=root/'results'/g['LABEL'];state={};pcis=['0000:%02x:00.0'%i for i in range(8)];common=list(range(32));uids=os.getuid()if hasattr(os,'getuid')else 154459
  def recording_run(argv,**kwargs):
   calls.append(argv);stdout='';rc=0
   if argv[0]=='scontrol'and argv[2]=='job':stdout='JobId=910371 NodeList=a02r3n[15-16]'
   elif argv[:3]==['scontrol','show','hostnames']:stdout='\n'.join(NODES)
   elif argv[0]=='squeue':stdout=''
   elif argv[0]=='sacct':stdout=gpu+'|'+g['LABEL']+'|COMPLETED|0:0|3\n'
   elif argv[0]=='scancel':signals.append(argv)
   elif argv[0]=='srun':stdout='\n'.join(json.dumps(dict(node=n,physical_PCI=pcis,CPU_only=True,kfd_pids=[],expected_owned_PIDs_still_live=[],ports_bindable={'43843':True,'43844':True}))for n in NODES)
   else:raise AssertionError('Unexpected control leaf '+repr(argv))
   return types.SimpleNamespace(returncode=rc,stdout=stdout,stderr='')
  class P:
   pid=os.getpid();returncode=None
   def __init__(self,argv,**kwargs):
    state['epoch']=float(next(x for x in argv if x.startswith('GLM470_EPOCH=')).split('=',1)[1]);self.argv=argv
    for node in range(2):
     owners=[]
     for dev in range(8):
      rank=node*8+dev;r=dict(global_rank=rank,device=dev,node=NODES[node],PID=9000+rank,UID=uids,owner_TID=9000+rank,startticks=str(8000+rank),completed=True,passed=True,normal_close=True,Runtime_weak_expired_CABI_assertion=True,actual_GPU=gpu,gpu_epoch=state['epoch'],entry_CPUmask=common,after_create_CPUmask=common,after_probe_CPUmask=common,SDK_first_submit_qualification=False,physical_PCI=pcis[dev],proof=dict(TP_SUM=136,TP_order=list(range(1,17))),Model_READY=False,weights=False,cache_allocated=False,HTTP=0,owned_socket_inodes=[],native_phase_records=[dict(rank=rank,phase=name,steady_ns=100+i)for i,name in enumerate(['allreduce-submit-begin','allreduce-submit-return','allgather-submit-begin','allgather-submit-return','original-sync-begin','original-sync-return'])])
      if corrupt and rank==15:r['proof']['TP_SUM']=135
      (out/('owner-%02d.json'%rank)).write_text(json.dumps(r));owners.append(dict(global_rank=rank,PID=r['PID'],startticks=r['startticks']))
     (out/('worker-%d.json'%node)).write_text(json.dumps(dict(completed=True,passed=True,actual_GPU=gpu,gpu_epoch=state['epoch'],owners=owners,rounds=[dict(phase='created',digest='a'*64),dict(phase='proved',digest='b'*64)])))
   def poll(self):return self.returncode
   def wait(self,timeout=None):
    if cancel and self.returncode is None:raise g['subprocess'].TimeoutExpired(self.argv,timeout)
    self.returncode=0;return 0
  def killpg(pid,sig):assert pid==P.pid;signals.append(dict(PGID=pid,signal=sig));P.returncode=-15
  read_text=Path.read_text
  def read_text_leaf(path,*args,**kwargs):
   if path.as_posix()=='/proc/'+str(P.pid)+'/stat':return ' '.join(['0']*21+['123456'])
   return read_text(path,*args,**kwargs)
  old=(g['R'],g['OUT'],g['verify']);g['R']=root;g['OUT']=out;g['verify']=lambda _:({'source_manifest_SHA256':'recording'},{});
  try:
   with M.patch.dict(os.environ,{'GLM470_GPU_MF':'recording','SLURM_JOB_ID':'910407','SLURM_STEP_ID':'9999'}),M.patch.object(g['subprocess'],'run',recording_run),M.patch.object(g['subprocess'],'Popen',P),M.patch.object(Path,'read_text',read_text_leaf),M.patch.object(g['os'],'getuid',lambda:uids,create=True),M.patch.object(g['os'],'killpg',killpg,create=True):rc=fn()
   q=json.loads((root/'results'/(g['LABEL']+'-watch.json')).read_bytes());assert q['completed']and q['passed']==(not corrupt and not cancel),q
   if not corrupt and not cancel:assert q['normal16']and len(q['owners'])==16 and len({(x['node'],x['physical_PCI'])for x in q['owners']})==16
   if cancel:assert q['forced_cleanup']and signals
   assert (rc==0)==q['passed'];return dict(passed=True,corrupt=corrupt,cancel=cancel,actual_controller_entry=True,Slurm_leaf='recording',device_leaf='recording',GPU=False,Model=False,source_requests=0,recording_control_calls=len(calls))
  finally:g['R'],g['OUT'],g['verify']=old;sys.path.pop(0)
if __name__=='__main__':print(json.dumps(dict(passed=True,cases=[case(),case(corrupt=True),case(cancel=True)],GPU=False)))
