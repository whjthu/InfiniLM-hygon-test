"""Actual loader/schedule/controller bodies; explicitly recording Native/Slurm leaves."""
from pathlib import Path
import tempfile,json,subprocess,sys,time
from unittest import mock
import controller
from consumer import recording_contract,NativeLeaf
from checkpoint_loader import check_recording

def controller_case(kind):
 real_popen=subprocess.Popen;real_sleep=time.sleep;calls=[];state=dict(submitted=False,transient=False,scancel=0)
 with tempfile.TemporaryDirectory(prefix='r398-controller-')as name:
  root=Path(name)
  for d in ['logs','results']:(root/d).mkdir()
  def popen(args,**kw):
   calls.append((args,kw));assert args==controller.gpu_command(root);assert kw['env']['GLM398_GPU_SOURCE_SHA256']=='a'*64 and float(kw['env']['GLM398_GPU_START'])>0
   state['submitted']=True
   p=real_popen([sys.executable,'-c','raise SystemExit('+str(2 if kind=='nonzero'else 0)+')'],stdin=subprocess.DEVNULL,stdout=kw['stdout'],stderr=kw['stderr'],start_new_session=True)
   p.wait(timeout=10)  # actual CPU child terminal; no synthetic live subprocess
   for rank in range(8):
    q=dict(completed=True,normal=True,normal_destroy=True,Runtime_weak_expired=True,Torch_after_destroy=True,policy_closed=True,measurement_calls=8,warm_calls=4,directV_diagnostics=2,actual_step='910371.999999',rank=rank,GPU_source_manifest_SHA256='a'*64,PID=398000+rank,physical_PCI='recording-pci-'+str(rank),Native_executed=False)
    (root/'results'/('GLM398-910371.999999-rank%d.json'%rank)).write_text(json.dumps(q))
   (root/'logs'/(controller.LABEL+'-910371.999999.log')).write_text('GLM398_ALL8_NORMAL\n'+''.join('GLM398_NORMAL %d\n'%r for r in range(8)))
   return p
  def output(args,**kw):
   if args[:3]==['scontrol','show','job']:return 'UserId=aclsylqidf(123) JobState=RUNNING NodeList=a02r3n03'
   if args[:3]==['scontrol','show','hostnames']:return 'a02r3n03\n'
   if args[0]=='sacct':
    if not state['submitted']:return ''
    if kind=='transient'and not state['transient']:state['transient']=True;raise subprocess.TimeoutExpired(args,30)
    return '910371.999999|'+controller.LABEL+'|'+('FAILED|2:0'if kind=='nonzero'else'COMPLETED|0:0')+'|a02r3n03\n'
   raise AssertionError(args)
  def run(args,**kw):
   if args[0]=='scancel':state['scancel']+=1;return subprocess.CompletedProcess(args,0,'','')
   assert args[0]=='srun'and '--gres=none'in args
   return subprocess.CompletedProcess(args,0,json.dumps(dict(passed=True,CPU_only=True,Native_executed=False))+'\n','')
  with mock.patch.object(controller,'verify',return_value={}),mock.patch.object(controller.subprocess,'Popen',side_effect=popen),mock.patch.object(controller.subprocess,'check_output',side_effect=output),mock.patch.object(controller.subprocess,'run',side_effect=run),mock.patch.object(controller.time,'sleep',side_effect=lambda _:real_sleep(.01)),mock.patch.object(controller.signal,'SIGKILL',9,create=True),mock.patch.object(controller.os,'killpg',side_effect=ProcessLookupError,create=True):
   q=controller.run(root,dict(publication_commit='b'*40,GPU_source_manifest_SHA256='a'*64,execute_once=True))
  assert len(calls)==1 and q['actual_step']=='910371.999999'and q['completed']
  assert q['passed']==(kind!='nonzero')and state['scancel']==0
  if kind=='transient':assert len(q['transient_observation_errors'])==1 and q['transient_observation_errors'][0]['original_start']==q['original_start']
  try:controller.start(root,'a'*64,popen=popen)
  except FileExistsError:pass
  else:raise AssertionError('duplicate launch log accepted')
  assert len(calls)==1
  return dict(kind=kind,actual_CPU_Popen=True,actual_controller_body=True,passed=q['passed'],Native=False,Slurm=False,submissions=1,scancel=state['scancel'],duplicate_rejected=True)

def destroy_failure_contract():
 calls=[]
 class Lib:
  def glm_attention_destroy_r398(self,*a):calls.append(a);return 1
  def glm_attention_error_r398(self):return b'recording destroy failure'
 leaf=NativeLeaf.__new__(NativeLeaf);leaf.lib=Lib();leaf.ctx=123;leaf.closed=False;leaf.destroy_attempted=False
 try:leaf.close()
 except RuntimeError:pass
 else:raise AssertionError('missing destroy failure')
 leaf.close();assert len(calls)==1 and not leaf.closed and leaf.destroy_attempted
 return dict(passed=True,actual_close_body=True,failed_destroy_not_reissued=True,Native=False)
def run():
 return dict(passed=True,schedule=recording_contract(),loader=check_recording(),destroy_failure=destroy_failure_contract(),controller=[controller_case(k)for k in ['normal','transient','nonzero']],GPU=0,Native_dlopen=False,checkpoint_payload_IO=False,actual_Slurm=False,numerical_qualification=False)
if __name__=='__main__':print(json.dumps(run()))
