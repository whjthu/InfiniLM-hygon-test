"""One CPU-only 17node idle snapshot. No GPU/API imports or daemon/poller."""
import concurrent.futures,json,os,re,subprocess,time
PARENTS={'910371':4,'910374':4,'910376':4,'910380':4,'910407':1}
ZONES={'910371':'GLM','910374':'DeepSeek','910376':'DeepSeek','910380':'Kimi','910407':'control'}
PROBE=r"""from pathlib import Path
import json,os,socket,time
node=socket.gethostname().split('.')[0];pci=sorted({p.resolve().name for p in Path('/sys/class/drm').glob('renderD*/device')});assert len(pci)==8
kfd=Path('/sys/class/kfd/kfd/proc');assert kfd.is_dir();k=[p.name for p in kfd.iterdir()if p.name.isdigit()];matching=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit()or int(p.name)==os.getpid():continue
 try:a=(p/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if any(b'full-model-entry-v2/owner.py'in x or b'full-model-entry-v2/control.py'in x or b'deepseek_v4_native'in x or b'GPU_run.py'in x for x in a):matching.append(int(p.name))
ports=[]
for port in [43843,43844]:
 with socket.socket()as s:
  try:s.bind(('0.0.0.0',port));ports.append(True)
  except OSError:ports.append(False)
print(json.dumps(dict(node=node,PCIs=pci,KFD_clean=not k,PID_clean=not matching,ports_clean=all(ports),membership_valid=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],observed_epoch=time.time())))
"""
def main():
 label='bw1000-tuning-idle-snapshot-'+str(time.time_ns())
 z=subprocess.run(['squeue','-h','-s','-j',','.join(PARENTS),'-o','%i|%j|%N'],capture_output=True,text=True,timeout=10);assert z.returncode==0
 active={};allocation={}
 for line in z.stdout.splitlines():
  step,name,hosts=line.split('|');h=subprocess.run(['scontrol','show','hostnames',hosts],capture_output=True,text=True,timeout=8);assert h.returncode==0
  # batch/extern are original owned allocation infrastructure, not GPU tasks.
  # KFD/proc/ports remain checked; all numbered or unknown steps block placement.
  store=allocation if re.fullmatch(r'(910371|910374|910376|910380|910407)[.](batch|extern)',step) else active
  for node in h.stdout.splitlines():store.setdefault(node,[]).append(step)
 env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_'))};env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
 def inspect(parent):
  n=PARENTS[parent];a=['srun','--jobid='+parent,'--overlap','--exact','--immediate=10','--label','--cpu-bind=none','-N'+str(n),'-n'+str(n),'--ntasks-per-node=1','-c1','--gres=none','--mem=256M','-t','00:01:00','--job-name='+label,'python3','-B','-c',PROBE];q=subprocess.run(a,env=env,capture_output=True,text=True,timeout=45);assert q.returncode==0,q.stderr[-300:];rows=[json.loads(l.partition(':')[2])for l in q.stdout.splitlines()];assert len(rows)==n;return parent,rows
 with concurrent.futures.ThreadPoolExecutor(max_workers=5)as pool:packets=list(pool.map(inspect,PARENTS))
 nodes={}
 for parent,rows in packets:
  for r in rows:
   name=r.pop('node');r.update(zone=ZONES[parent],active_steps=active.get(name,[]),allocation_only_steps=allocation.get(name,[]));nodes[name]=r
 assert len(nodes)==17
 return dict(observed_epoch=min(r['observed_epoch']for r in nodes.values()),complete=True,total_cards=136,nodes=nodes,scope='CPU-only idle PCI/KFD/proc/ports snapshot, no GPU work, not utilization measurement')
if __name__=='__main__':print(json.dumps(main()))
