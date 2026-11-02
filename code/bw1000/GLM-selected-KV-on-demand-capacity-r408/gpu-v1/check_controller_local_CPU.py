"""Actual controller.main with recording subprocess leaves; no jobs/GPU/Native."""
from pathlib import Path
import tempfile,json,hashlib,types,sys,os,subprocess
from unittest.mock import patch
import controller as c
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def case(kind):
 with tempfile.TemporaryDirectory()as tmp:
  r=Path(tmp);s=r/'incoming';s.mkdir();(r/'results').mkdir();(r/'logs').mkdir();(s/'outer.sh').write_text('test-wrapper')
  lib=r/'fakeDSO';lib.write_bytes(b'NOT_A_REAL_DSO');cp=r/'CPU.json';cp.write_text(json.dumps(dict(completed=True,passed=True,library=str(lib),library_SHA256=sha(lib),fresh_M_before_compile=True,actual_HOST_contract=dict(passed=True),actual_CPU_consumer=dict(passed=True),source_manifest_SHA256='cpu-source')))
  mf=s/'GPU_MANIFEST.json';mf.write_text(json.dumps(dict(GPU_GO=True,source_SHA256={'outer.sh':sha(s/'outer.sh')},CPU_receipt=str(cp),CPU_receipt_SHA256=sha(cp),bridge_SHA256=sha(lib),CPU_source_manifest_SHA256='cpu-source')))
  for rank in range(8):
   q=dict(completed=True,passed=kind!='driver_fail',normal=True,actual_step='910371.1',rank=rank,GPU_source_manifest_SHA256=sha(mf),library_SHA256=sha(lib),normal_destroy=True,Runtime_weak_expired=True,Torch_after_destroy=True,policy_closed=True,measurement_calls=8,warm_calls=4,directK_diagnostics=2,records=[{}for i in range(12)],PID=9000+rank,physical_PCI='fakePCI'+str(rank))
   (r/('results/GLM408-910371.1-rank%d.json'%rank)).write_text(json.dumps(q))
  clock=[10000.];state=dict(submitted=False,popen=0,lost=False,cancels=0)
  class Proc:
   pid=1234
   def wait(self,timeout):return 0
  class Sub:
   CalledProcessError=subprocess.CalledProcessError;TimeoutExpired=subprocess.TimeoutExpired
   DEVNULL=subprocess.DEVNULL;STDOUT=subprocess.STDOUT
   @staticmethod
   def check_output(args,**kw):
    if args[:3]==['scontrol','show','job']:return 'JobState=RUNNING UserId=aclsylqidf(1) NodeList=a02r3n16 '
    if args[:3]==['scontrol','show','hostnames']:return 'a02r3n16\n'
    if args[0]=='sacct':
     if not state['submitted']:return ''
     if kind=='lost'and not state['lost']:state['lost']=True;raise subprocess.TimeoutExpired(args,1)
     return '910371.1|GLM-selected-KV-on-demand-capacity-r408-v1|COMPLETED|0:0|a02r3n16\n'
    raise AssertionError(args)
   @staticmethod
   def run(args,**kw):
    if args[0]=='scancel':state['cancels']+=1;raise AssertionError('Unexpected cancel')
    assert args[0]=='srun'and'--gres=none'in args and'-c1'in args
    d=dict(node='a02r3n16',actual_step='910371.2',KFD_available=True,kfd_pids=[999]if kind=='busy'else[],matching_component_processes=[],port43882_listening=False)
    return types.SimpleNamespace(returncode=0,stdout=json.dumps(d)+'\n',stderr='')
   @staticmethod
   def Popen(args,**kw):
    state['popen']+=1;assert state['popen']==1 and'--gres=dcu:8'in args and'-c32'in args and'00:10:00'in args and'--kill-on-bad-exit=0'in args
    assert '--jobid=910371'in args and args[args.index('-w')+1]=='a02r3n16';assert kw['env']['GLM408_GPU_START']=='10000.0'
    state['submitted']=True;return Proc()
  def sleep(v):clock[0]+=v
  old=(c.R,c.S,c.W,c.subprocess,c.time)
  c.R=r;c.S=s;c.W=r/'results/watch.json';c.subprocess=Sub;c.time=types.SimpleNamespace(time=lambda:clock[0],monotonic=lambda:clock[0],sleep=sleep)
  try:
   with patch.dict(os.environ,{'SLURM_JOB_ID':'910407','SLURM_STEP_ID':'FAKE_CPU'},clear=False),patch.object(sys,'argv',['controller.py','--publication-commit','a'*40,'--manifest-sha',sha(mf)]):
    try:c.main()
    except AssertionError:assert kind in ['busy','driver_fail']
    else:assert kind in ['normal','lost']
   watch=json.loads(c.W.read_bytes());assert watch['GPU_max_s']==600 and state['cancels']==0
   if kind=='busy':assert state['popen']==0 and watch['GPU_submitted']is False
   elif kind=='driver_fail':assert state['popen']==1 and watch['passed']is False
   else:assert state['popen']==1 and watch['passed']is True and watch['actual_GPU']=='910371.1'
   return dict(kind=kind,popen=state['popen'],no_cancel=True,passed=True)
  finally:c.R,c.S,c.W,c.subprocess,c.time=old
if __name__=='__main__':print(json.dumps(dict(passed=True,actual_controller_main=[case(k)for k in ['normal','lost','busy','driver_fail']],jobs_submitted=0,GPU_initialized=False,Native_dlopen=False)))
