"""Actual controller.main with recording subprocess leaves; no jobs/GPU/Native."""
from pathlib import Path
import tempfile,json,hashlib,types,sys,os,subprocess
from unittest.mock import patch
import controller as c
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def case(kind):
 with tempfile.TemporaryDirectory()as tmp:
  r=Path(tmp);s=r/'incoming';s.mkdir();(r/'results').mkdir();(r/'logs').mkdir();(s/'outer.sh').write_text('test-wrapper')
  lib=r/'fakeDSO';lib.write_bytes(b'NOT_A_REAL_DSO');native=r/'fakeNative';native.write_bytes(b'NOT_NATIVE');meta=r/'fakeMetadata';meta.write_bytes(b'NOT_EXECUTED')
  cp=r/'CPU.json';cp.write_text(json.dumps(dict(completed=True,passed=False,actual_CPU='910407.851')))
  closure=s/'CPU_ARTIFACT_CLOSURE.json';closure.write_text(json.dumps(dict(original_CPU_producer_passed=False,read_only_verifier_passed=True,existing_artifact_CPU_math_and_link_checks_passed=True,original_validate_pass_count=13,known_CPU_backend_classified_cases=['negative_underflow'],library=dict(SHA256=sha(lib)),private_DSO_DT_NEEDED=[str(native)])))
  objects=[]
  for name in ['private_moe','adapter','router_kernel']:
   obj=r/(name+'.o');obj.write_bytes(name.encode());md=r/(name+'.d');md.write_bytes((name+'MD').encode());objects.append(dict(object=str(obj),object_SHA256=sha(obj),MD_path=str(md),MD_SHA256=sha(md)))
  mf=s/'GPU_MANIFEST.json';mf.write_text(json.dumps(dict(GPU_GO=True,Model_GO=False,source_SHA256={'outer.sh':sha(s/'outer.sh'),'CPU_ARTIFACT_CLOSURE.json':sha(closure)},CPU_original_receipt=str(cp),CPU_original_receipt_SHA256=sha(cp),CPU_existing_artifact_closure_SHA256=sha(closure),bridge=str(lib),bridge_SHA256=sha(lib),Native=str(native),Native_SHA256=sha(native),metadata_executable=str(meta),metadata_executable_SHA256=sha(meta),compiled_objects=objects)))
  q=dict(completed=True,passed=kind!='driver_fail',normal=True,owners=[dict(normal_close=True)for i in range(8)])
  (r/'results/Deep419-MoE-910374.1.json').write_text(json.dumps(q))
  clock=[10000.];state=dict(submitted=False,popen=0,lost=False,cancels=0)
  class Proc:
   pid=1234
   def wait(self,timeout):return 0
  class Sub:
   CalledProcessError=subprocess.CalledProcessError;TimeoutExpired=subprocess.TimeoutExpired
   DEVNULL=subprocess.DEVNULL;STDOUT=subprocess.STDOUT
   @staticmethod
   def check_output(args,**kw):
    if args[:3]==['scontrol','show','job']:return 'JobState=RUNNING UserId=aclsylqidf(1) NodeList=a02r4n04 '
    if args[:3]==['scontrol','show','hostnames']:return 'a02r4n04\n'
    if args[0]=='sacct':
     if not state['submitted']:return ''
     if kind=='lost'and not state['lost']:state['lost']=True;raise subprocess.TimeoutExpired(args,1)
     return '910374.1|Deep-learned-router-suffix-r419-v1|COMPLETED|0:0|a02r4n04\n'
    raise AssertionError(args)
   @staticmethod
   def run(args,**kw):
    if args[0]=='scancel':state['cancels']+=1;raise AssertionError('Unexpected cancel')
    assert args[0]=='srun'and'--gres=none'in args and'-c1'in args
    d=dict(node='a02r4n04',actual_step='910374.2',KFD_available=True,kfd_pids=[999]if kind=='busy'else[],matching_component_processes=[],port43882_listening=False)
    return types.SimpleNamespace(returncode=0,stdout=json.dumps(d)+'\n',stderr='')
   @staticmethod
   def Popen(args,**kw):
    state['popen']+=1;assert state['popen']==1 and'--gres=dcu:8'in args and'-c32'in args and'00:10:00'in args and'--kill-on-bad-exit=0'in args
    assert '--jobid=910374'in args and args[args.index('-w')+1]=='a02r4n04';assert kw['env']['DEEP419_ABSOLUTE_DEADLINE']=='10590.0'
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
   else:assert state['popen']==1 and watch['passed']is True and watch['actual_GPU']=='910374.1'
   return dict(kind=kind,popen=state['popen'],no_cancel=True,passed=True)
  finally:c.R,c.S,c.W,c.subprocess,c.time=old
if __name__=='__main__':print(json.dumps(dict(passed=True,actual_controller_main=[case(k)for k in ['normal','lost','busy','driver_fail']],jobs_submitted=0,GPU_initialized=False,Native_dlopen=False)))
