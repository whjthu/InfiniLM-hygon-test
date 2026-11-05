"""One180s CPUkeeper; fresh then immutable GPUepoch120, independent post."""
from pathlib import Path
import json,os,re,subprocess,time
from support import D,R,LABEL,NODE,OUT,save,verify,H

def command(label,seconds,cards,script,cores=1):
 return ['srun','--jobid=910374','--overlap','--exact','--immediate=10','--cpu-bind=none','-N1','-n1','-c'+str(cores),'--mem='+('32768M'if cards else'8192M'),'--gres='+('dcu:8'if cards else'none'),'-w',NODE,'-t',str(seconds),'--job-name='+label,'--output='+str(R/'logs'/(label+'-%J.log')),*script]
def main():
 start=time.monotonic();m,cpu=verify();assert not OUT.exists();watch=R/'results'/(LABEL+'-watch.json');assert not watch.exists();q=dict(completed=False,passed=False,CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=None,epoch=None,weights=False,Model=False,HTTP=0)
 def run(a,seconds):
  assert time.monotonic()-start+seconds<178;z=subprocess.run(a,capture_output=True,text=True,timeout=seconds);return dict(argv=a,rc=z.returncode,stdout=z.stdout,stderr=z.stderr)
 try:
  q['fresh']=run(command(LABEL+'-before','00:00:20',False,['python3','-B',str(D/'idle.py')]),25);save(watch,q);assert q['fresh']['rc']==0
  epoch=time.time();q['epoch']=epoch;env=dict(os.environ,HC_META_EPOCH=str(epoch),HC_META_OUTPUT='/refactor/results/'+LABEL,WAVE_INSIDE_R2323='Deep-C1-gap-enclosure-r467/hc-metadata-GPU-v2/inside.sh');outer=D/'outer.sh';a=command(LABEL,'00:02:00',True,['env','HC_META_EPOCH='+str(epoch),'HC_META_OUTPUT=/refactor/results/'+LABEL,'WAVE_INSIDE_R2323='+env['WAVE_INSIDE_R2323'],'bash',str(outer),'probe'],32);log=(R/'logs'/(LABEL+'-srun.log')).open('xb');p=subprocess.Popen(a,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);log.close();q.update(owned_PID=p.pid,owned_PGID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21]);save(watch,q)
  try:q['GPU_srun_rc']=p.wait(timeout=max(1,epoch+125-time.time()))
  except subprocess.TimeoutExpired:os.killpg(p.pid,15);q['forced_TERM']=True;p.wait(timeout=5);raise
  if(OUT/'worker.json').exists():w=json.loads((OUT/'worker.json').read_bytes());q['GPU']=w['actual_GPU'];q['worker_SHA256']=H(OUT/'worker.json');q['worker_passed']=w['passed'];assert w['passed']and q['GPU_srun_rc']==0
  else:raise AssertionError('Missing original worker report')
  q['passed']=True
 except BaseException as e:q['error']=repr(e)
 finally:
  expected=[]
  if(OUT/'worker.json').exists():
   data=json.loads((OUT/'worker.json').read_bytes());expected=[dict(PID=r['PID'],startticks=r['startticks'])for r in data.get('owners',[])]
  postenv=['env','DEEP467_PROFILE_EXPECTED_PIDS='+json.dumps({NODE:expected},separators=(',',':'))]
  try:q['post']=run(command(LABEL+'-post','00:00:20',False,postenv+['python3','-B',str(D/'idle.py')]),25);q['passed']=q['passed']and q['post']['rc']==0
  except BaseException as e:q['post_error']=repr(e);q['passed']=False
  q.update(completed=True,seconds=time.monotonic()-start);save(watch,q)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
