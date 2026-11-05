"""Exactly8localowners pernode,16globalowners total; no retries."""
from pathlib import Path
import json,os,sys,subprocess,time,socket
from shared import D,OUT,NODES,save,verify

def main():
 node=int(os.environ['SLURM_PROCID']);assert node in (0,1)and socket.gethostname().split('.')[0]==NODES[node];epoch=float(os.environ['GLM470_EPOCH']);verify(os.environ['GLM470_GPU_MF']);allowed=sorted(os.sched_getaffinity(0));assert len(allowed)>=32;children=[];q=dict(completed=False,passed=False,node=NODES[node],node_index=node,actual_GPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],gpu_epoch=epoch,common_CPUmask=allowed,owners=[])
 try:
  for d in range(8):
   gid=node*8+d
   with(OUT/('owner-%02d.log'%gid)).open('xb')as f:p=subprocess.Popen([sys.executable,'-B',str(D/'owner.py'),str(node),str(d)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   q['owners'].append(dict(global_rank=gid,PID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21]));children.append(p);save(OUT/('worker-%d.json'%node),q)
  while any(p.poll()is None for p in children):
   if any(p.poll()not in(None,0)for p in children):raise RuntimeError('Owned component childfailed; no recreate')
   if time.time()>=epoch+150:raise TimeoutError('Original150worker deadline')
   time.sleep(.1)
  for p,r in zip(children,q['owners']):
   assert p.returncode==0;f=OUT/('owner-%02d.json'%r['global_rank']);z=json.loads(f.read_bytes());assert z['completed']and z['passed']and z['normal_close']and z['PID']==r['PID']and z['startticks']==r['startticks']and z['entry_CPUmask']==allowed;r['returncode']=0
  q['passed']=True
 finally:
  for p in children:
   if p.poll()is None:p.terminate();q['forced_TERM']=True
  for p in children:
   try:p.wait(timeout=2)
   except subprocess.TimeoutExpired:p.kill();p.wait(timeout=2);q['forced_KILL']=True
  q['completed']=True;save(OUT/('worker-%d.json'%node),q)
if __name__=='__main__':main()
