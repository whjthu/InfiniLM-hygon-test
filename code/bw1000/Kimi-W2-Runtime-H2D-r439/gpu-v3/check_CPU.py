"""Actual parent/stdio/filesystem; pure Python child recordings, no Runtime."""
from pathlib import Path
import contextlib,hashlib,importlib.util,io,json,os,subprocess,sys,tempfile,time
from unittest.mock import patch
import fixed_children_r439 as fixed
import GPU_run as driver
D=Path(__file__).resolve().parent
real_popen=subprocess.Popen
def metadata(root):
 result={}
 for role in ['baseline','candidate']:
  exe=root/(role+'.exe');lib=root/('libinfinilm_runtime-'+role+'.so');exe.write_bytes(role.encode());lib.write_bytes(role.encode())
  result[role]=dict(path=str(exe),SHA256=driver.sha(exe),Runtime=dict(path=str(lib),SHA256=driver.sha(lib)))
 p=root/'metadata.json';p.write_text(json.dumps(result));return result,p
def leaf(argv,**kwargs):return real_popen([sys.executable,'-B','-u',str(D/'recording_child_CPU.py'),*argv[1:]],**kwargs)
def case(kind):
 with tempfile.TemporaryDirectory(prefix='kimi439-stdio-')as td:
  root=Path(td);assert root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve());(root/'results').mkdir();providers,mp=metadata(root)
  cp=root/'receipt.json';cp.write_text(json.dumps(dict(passed=True,actual_CPU='recording',Runtime_executions=0,executables=providers)))
  mf=root/'GPU_SOURCE_MANIFEST.json';mf.write_text(json.dumps(dict(files_SHA256={},CPU_receipt=str(cp),CPU_receipt_SHA256=driver.sha(cp),CPU_actual='recording')))
  env=dict(SLURM_JOB_ID='910371',SLURM_STEP_ID='1',R439_GPU_START=str(time.time()),R439_GPU_SOURCE_SHA256=driver.sha(mf),R439_TEST_METADATA=str(mp),R439_TEST_MODE='normal')
  original_send=fixed.Peer.send
  def bad_width(peer,item):
   if item['seq']==1:item=dict(item,width=1)
   return original_send(peer,item)
  def cancelled():
   yield fixed.plan()[0]
   raise KeyboardInterrupt('recorded parent cancellation')
  with patch.object(driver,'D',root),patch.object(driver,'R',root),patch.dict(os.environ,env),patch.object(driver.os,'sched_getaffinity',return_value={0,1,2,3},create=True),patch.object(driver.os,'sched_setaffinity',create=True),contextlib.redirect_stdout(io.StringIO()):
   with contextlib.ExitStack()as stack:
    if kind=='bad_width':stack.enter_context(patch.object(fixed.Peer,'send',bad_width))
    if kind=='cancel':stack.enter_context(patch.object(driver,'plan',cancelled))
    rc=driver.main(popen=leaf)
  q=json.loads((root/'results/Kimi439-910371.1-main.json').read_text());assert q['completed']and len(q['actors'])==2
  assert all(v.get('rc')is not None and v.get('reader_stopped')for v in q['process_cleanup'].values())
  if kind=='normal':assert rc==0 and q['passed']and q['work_qualified']==38 and q['normal_Runtime_closes']==2 and q['stable_component_gain']
  else:assert rc==1 and not q['passed']and q['work_qualified']==1 and q['error']
  return dict(case=kind,passed=True,work_qualified=q['work_qualified'],normal_Runtime_closes=q['normal_Runtime_closes'],actual_stdio=True,CXX_Runtime_device_leaf='recording',owned_PIDs_reaped=True)
def startup(kind):
 with tempfile.TemporaryDirectory(prefix='kimi439-startup-')as td:
  root=Path(td);assert root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve());providers,mp=metadata(root);env=dict(os.environ,R439_TEST_METADATA=str(mp),R439_TEST_MODE=kind)
  p=fixed.Peer(providers['baseline']['path'],'baseline',root/'actor',time.time()+2,env,popen=leaf);caught=None
  try:p.receive(time.time()+(.15 if kind=='timeout'else 1.5))
  except(TimeoutError,EOFError)as e:caught=type(e).__name__
  finally:r=p.finish(time.time()+.15)
  assert caught==('TimeoutError'if kind=='timeout'else'EOFError')and r['rc']is not None and r['reader_stopped']
  return dict(case=kind,passed=True,actual_stdio=True,owned_PID_reaped=True,forced_kill=r['forced_kill'],CXX_Runtime_device_leaf='recording')
if __name__=='__main__':print(json.dumps(dict(passed=True,cases=[case('normal'),case('bad_width'),case('cancel'),startup('timeout'),startup('EOF')],Runtime=False,GPU=False,Native=False),indent=2))
