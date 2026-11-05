"""One CPU-only reproduction of the original16marker save/wait, no SDK imports."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time,socket,runpy,sys,re,traceback
D=Path(__file__).resolve().parent;R=Path('/refactor')if str(D).startswith('/refactor/')else Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');LABEL='GLM-r470-marker-barrier-CPU-v1';OUT=R/'results'/LABEL;H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();NODES=['a02r3n15','a02r3n16']
def parent():
 p=D.parent/'device-v1/shared.py';assert H(p)=='4c14563023e21292dcadad5402a5d68c9f2c5551551f6db43e2c6aad665b872c';return runpy.run_path(str(p))
def save(p,q):
 t=p.with_name(p.name+'.tmp-'+str(os.getpid()));t.write_text(json.dumps(q,indent=2)+'\n');os.replace(str(t),str(p))
def verify():
 claim=json.loads((D/'CPU_RESOURCE_CLAIM.json').read_bytes());assert claim['CPU_GO']and not claim['GPU_GO'];assert H(D/'SOURCE_MANIFEST.json')==claim['source_manifest_SHA256'];assert all(H(D/n)==h for n,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items());return claim

def child(node,device):
 api=parent();original_save,original_wait=api['save'],api['wait'];epoch=float(os.environ['GLM470_BARRIER_EPOCH']);rank=node*8+device;report=OUT/('owner-%02d.json'%rank);q=dict(node=NODES[node],node_index=node,rank=rank,device=device,PID=os.getpid(),startticks=Path('/proc/self/stat').read_text().split()[21],epoch=epoch,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,SDK_initialized=False,GPU=False,source_shared_SHA256=H(D.parent/'device-v1/shared.py'),waits=[],phase='before_save',passed=False,completed=False)
 try:
  assert socket.gethostname().split('.')[0]==NODES[node];save(report,q);p=OUT/('created-%02d.json'%rank);t=time.monotonic();original_save(p,dict(global_rank=rank,epoch=epoch));q['save_seconds']=time.monotonic()-t;q['phase']='saved';save(report,q)
  for i in range(16):
   p=OUT/('created-%02d.json'%i);q['phase']='wait_%02d'%i;save(report,q);t=time.monotonic();value=original_wait(p,epoch);seconds=time.monotonic()-t;assert value==dict(global_rank=i,epoch=epoch);q['waits'].append(dict(rank=i,path=str(p),seconds=seconds,exists_after=True,bytes=p.stat().st_size,SHA256=H(p)));save(report,q)
  q['phase']='all16_read';q['passed']=True
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:q['completed']=True;save(report,q)

def worker():
 verify();node=int(os.environ['SLURM_PROCID']);assert node in (0,1)and socket.gethostname().split('.')[0]==NODES[node];parent();epoch=float(os.environ['GLM470_BARRIER_EPOCH']);children=[];q=dict(node=NODES[node],actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],epoch=epoch,owners=[],completed=False,passed=False)
 try:
  for dev in range(8):
   with(OUT/('owner-%02d.log'%(node*8+dev))).open('xb')as f:p=subprocess.Popen([sys.executable,'-B',str(D/'barrier.py'),'child','--node',str(node),'--device',str(dev)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   children.append(p);q['owners'].append(dict(rank=node*8+dev,PID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21]));save(OUT/('worker-%d.json'%node),q)
  for p in children:assert p.wait(timeout=max(.1,epoch+155-time.time()))==0
  q['passed']=True
 finally:
  for p in children:
   if p.poll()is None:p.terminate();q['forced_TERM']=True
  for p in children:
   try:p.wait(timeout=2)
   except subprocess.TimeoutExpired:p.kill();p.wait(timeout=2);q['forced_KILL']=True
  q['completed']=True;save(OUT/('worker-%d.json'%node),q)

def launch(publication):
 verify();assert re.fullmatch('[0-9a-f]{40}',publication);assert str(D).startswith('/public/home/');intent=R/'results'/(LABEL+'-intent.json');launch=R/'results'/(LABEL+'-launch.json');assert not any(p.exists()for p in [OUT,intent,launch]);z=subprocess.run(['sacct','-j','910371','-S','2026-11-05','-n','-P','-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert z.returncode==0 and not any(LABEL in line for line in z.stdout.splitlines());epoch=time.time();a=['srun','--jobid=910371','--overlap','--exact','--immediate=10','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c1','--mem=1024M','--gres=none','-w',','.join(NODES),'-t','00:03:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J.log')),'env','GLM470_BARRIER_EPOCH='+str(epoch),'WAVE_INSIDE_R2323=GLM952-global-TP16-component-r470/barrier-CPU-v1/inside.sh','bash',str(D/'outer.sh'),'probe']
 with intent.open('x')as f:json.dump(dict(publication=publication,source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),claim_SHA256=H(D/'CPU_RESOURCE_CLAIM.json'),HOST=socket.gethostname(),epoch=epoch,argv=a,GPU=False),f)
 OUT.mkdir(exist_ok=False)
 with(R/'logs'/(LABEL+'-srun.log')).open('xb')as f:p=subprocess.Popen(a,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 q=dict(PID=p.pid,PGID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],HOST=socket.gethostname(),epoch=epoch,GPU=False,intent=str(intent));save(launch,q);print(json.dumps(q))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','worker','child']);p.add_argument('--publication');p.add_argument('--node',type=int);p.add_argument('--device',type=int);a=p.parse_args()
 if a.mode=='launch':launch(a.publication)
 elif a.mode=='worker':worker()
 else:child(a.node,a.device)
