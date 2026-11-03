"""Only new Python main/filesystem/stop glue; executable results are recordings."""
from pathlib import Path
import contextlib,hashlib,importlib.util,io,json,os,tempfile,time
from unittest.mock import patch
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r439_real_driver',D/'GPU_run.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
def check(fail):
 with tempfile.TemporaryDirectory(prefix='kimi439-')as td:
  root=Path(td);assert root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
  (root/'results').mkdir();providers={}
  for name in ['baseline','candidate']:
   exe=root/(name+'.exe');lib=root/(name+'Runtime.so');exe.write_bytes(name.encode());lib.write_bytes((name+'Runtime').encode())
   providers[name]=dict(path=str(exe),SHA256=driver.sha(exe),Runtime=dict(path=str(lib),SHA256=driver.sha(lib)))
  cp=root/'receipt.json';cp.write_text(json.dumps(dict(passed=True,actual_CPU='recording',Runtime_executions=0,executables=providers)))
  manifest=root/'GPU_SOURCE_MANIFEST.json';manifest.write_text(json.dumps(dict(files_SHA256={},CPU_receipt=str(cp),CPU_receipt_SHA256=driver.sha(cp),CPU_actual='recording')))
  class Leaf:
   number=0
   def __init__(self,argv,**kwargs):
    self.argv=argv;self.env=kwargs['env'];Leaf.number+=1;self.pid=10000+Leaf.number;self.returncode=0
   def communicate(self,timeout=None):
    exe,operation,width,result=self.argv;width=int(width);arm='candidate'if self.env['INFINILM_CPU_ROW_COPY_R436']=='1'else'baseline';bad=fail and arm=='candidate'
    a=dict(passed=not bad,Runtime_expired=True,on=arm=='candidate',operation=operation,width=width,Runtime_provider=providers[arm]['Runtime']['path'],maps=providers[arm]['Runtime']['path'],device=dict(physical_PCI='0000:00:00.0',CU=80,total_memory=64*1024**3),cases=36 if operation=='check'else 896,offsets=list(range(32)),stats_delta=[33,0,0,0,0],hits_delta=[32,1]if arm=='candidate'else[0,0],main_wall_s=.001)
    if operation=='bench':a.update(expert_copy_calls=896,actual_H2D_bytes=896*3584*width,allocation_in_timed_wall=False,timed_stats_delta=[896,896*3584,896*3584*width,0,0],timed_hits_delta=[896,0]if arm=='candidate'else[0,0],full_gather_H2D_sync_wall_s=.5 if arm=='candidate'else 1.)
    if bad:a['error']='recorded firstON byte failure';self.returncode=1
    Path(result).write_text(json.dumps(a));return json.dumps(a),'recording C++ leaf, no Runtime/device'
   def kill(self):self.returncode=-9
  env=dict(SLURM_JOB_ID='910371',SLURM_STEP_ID='1',R439_GPU_START=str(time.time()),R439_GPU_SOURCE_SHA256=driver.sha(manifest))
  with patch.object(driver,'D',root),patch.object(driver,'R',root),patch.dict(os.environ,env),patch.object(driver.os,'sched_getaffinity',return_value={0,1,2,3},create=True),patch.object(driver.os,'sched_setaffinity',create=True),patch.object(driver.subprocess,'Popen',Leaf),contextlib.redirect_stdout(io.StringIO()):rc=driver.main()
  q=json.loads((root/'results/Kimi439-910371.1-main.json').read_text())
  assert q['completed']and len(q['records'])==(2 if fail else 38)and q['Runtime_processes_normal']==(1 if fail else 38)
  if fail:assert rc==1 and not q['passed']and q['records'][-1]['error']=='recorded firstON byte failure'and Path(q['records'][-1]['actual_result']['path']).is_file()
  else:assert rc==0 and q['passed']and q['all_normal']and q['stable_component_gain']and len(q['paired'])==2
  return dict(passed=True,scenario='firstON completed failure persisted before STOP'if fail else'actual main full38 plan/filesystem',CXX_Runtime_device_leaf='recording',records=len(q['records']))
if __name__=='__main__':print(json.dumps(dict(passed=True,cases=[check(False),check(True)],GPU=False,Native=False,Runtime=False),indent=2))
