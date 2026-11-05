"""Eight independent originalowner processes, no Model/HTTP/CCL."""
from pathlib import Path
import hashlib,json,os,subprocess,time
from support import D,H,verify,save

def validate_rank(z,r,allowed):
 assert z['completed']and z['passed']and z['normal_Runtime_expired']and z['rank']==r['rank']and len(z['frames'])==7;assert z['entry_CPUmask']==allowed;f0=z['frames'][0];assert f0['mask_observed_after_original_completion']and f0['actual_CPUmask_after_first_completed_frame']==r['expected_mask']and f0['owner_TID']==r['PID']and f0['owner_pthread'];r['actual_SDK_bound_mask']=f0['actual_CPUmask_after_first_completed_frame'];r['actual_owner_TID']=f0['owner_TID'];assert all(len(v['ops'])==12 and v['finite_CPUref_passed']for v in z['frames']);

def run(out,epoch):
 out=Path(out);out.mkdir(exist_ok=False);m,cpu=verify();allowed=sorted(os.sched_getaffinity(0));assert len(allowed)>=32;children=[];observed={};start=time.monotonic();deadline=epoch+100
 q=dict(completed=False,passed=False,actual_GPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],gpu_epoch=epoch,allowed_CPUs=allowed,owners=[],Model=False,weights=False,HTTP=0)
 def flush():
  q['map_observations']=observed;save(out/'worker.json',q)
 try:
  for rank in range(8):
   mask=allowed[rank*4:(rank+1)*4];report=out/('rank'+str(rank)+'.json');log=(out/('rank'+str(rank)+'.log')).open('xb')
   p=subprocess.Popen([cpu['executable'],str(rank),str(report)],stdout=log,stderr=subprocess.STDOUT);log.close();record=dict(rank=rank,PID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],expected_mask=mask,actual_mask=sorted(os.sched_getaffinity(p.pid)),report=str(report));# Entry must remain common until SDKfirstsubmit; C++selfrecord is authoritative.
   q['owners'].append(record);children.append((p,record));flush()
   # Observe loader maps immediately: fast noModel children may finish before
   # all8 Popen calls complete. This is startup observation, not timedmath.
   wanted=[cpu['executable'],*[x for x in cpu['parent_libraries']if not Path(x).name.startswith('libDeep_')]];until=min(deadline,time.time()+1)
   while p.poll()is None and time.time()<until:
    try:
     text=Path('/proc/'+str(p.pid)+'/maps').read_text();paths={x.split()[-1]for x in text.splitlines()if '/'in x and x.split()[-1].startswith('/')};paths={('/refactor'+x[len('/public/home/aclsylqidf/hygon-inference-refactor-20261014'):])if x.startswith('/public/home/aclsylqidf/hygon-inference-refactor-20261014/')else x for x in paths}
     if all(x in paths for x in wanted):observed[str(rank)]={x:H(x)for x in wanted};record['observed_live_mask']=sorted(os.sched_getaffinity(p.pid));break
    except(FileNotFoundError,ProcessLookupError):pass
    time.sleep(.002)

  while any(p.poll()is None for p,r in children):
   if time.time()>=deadline:raise TimeoutError('Original100s deadline')
   for p,r in children:
    if p.poll()is None:
     try:
      maps=Path('/proc/'+str(p.pid)+'/maps').read_text();paths=sorted({x.split()[-1]for x in maps.splitlines()if '/'in x and x.split()[-1].startswith('/')});required=[cpu['executable'],*[x for x in cpu['parent_libraries']if not Path(x).name.startswith('libDeep_')]];paths=[('/refactor'+x[len('/public/home/aclsylqidf/hygon-inference-refactor-20261014'):])if x.startswith('/public/home/aclsylqidf/hygon-inference-refactor-20261014/')else x for x in paths]
      if all(x in paths for x in required):observed[str(r['rank'])]={x:H(x)for x in required}
     except(FileNotFoundError,ProcessLookupError):pass
   time.sleep(.01)
  for p,r in children:
   r['returncode']=p.returncode;assert p.returncode==0;r['report_SHA256']=H(r['report']);z=json.loads(Path(r['report']).read_bytes());validate_rank(z,r,allowed);r['metadata']=z['metadata'];r['Mul_Torch8_hypothesis']=z['Mul_Torch8_hypothesis'];assert str(r['rank'])in observed,'Actual SDK maps not observed; do not qualify';r['actual_provider_maps']=observed[str(r['rank'])]
  q['passed']=True
 except BaseException as e:
  q['error']=repr(e);raise
 finally:
  # Only exact owned subprocess handles. Forced cleanup never passes.
  for p,r in children:
   if p.poll()is None:p.terminate();r['TERM']=True
  for p,r in children:
   try:p.wait(timeout=2)
   except subprocess.TimeoutExpired:p.kill();r['KILL']=True;p.wait(timeout=2)
  q.update(completed=True,seconds=time.monotonic()-start);flush()
 return q
if __name__=='__main__':run(os.environ['HC_META_OUTPUT'],float(os.environ['HC_META_EPOCH']))
