"""Finite16card component keeper; original180/150 epoch, independent post."""
from pathlib import Path
import os,sys,time,json,subprocess,re,traceback,signal
from shared import D,R,OUT,LABEL,NODES,H,save,verify
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
def clean_env():return{k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
def main():
 started=time.monotonic();limit=started+290;m,cpu=verify(os.environ['GLM470_GPU_MF']);watch=R/'results'/(LABEL+'-watch.json');assert not watch.exists()and not OUT.exists();q=dict(completed=False,passed=False,GPU=None,CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_seconds=180,worker_seconds=150,Model=False,weights=False,HTTP=0,submissions=[]);p=None;step=None;expected={}
 def persist():save(watch,q)
 def run(a,seconds):
  assert time.monotonic()+seconds<=limit;z=subprocess.run(a,env=clean_env(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=seconds);return dict(argv=a,rc=z.returncode,stdout=z.stdout,stderr=z.stderr)
 def account():
  a=['sacct','-j','910371','-S','2026-11-05','-n','-P','-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw'];z=run(a,12);assert z['rc']==0;rows=[v.split('|')for v in z['stdout'].splitlines()if v.split('|')[1]==LABEL];assert len(rows)<=1;return rows[0]if rows else None
 def command(label,gpu,script):return ['srun','--jobid=910371','--overlap','--exact','--immediate=10','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c'+('32'if gpu else'1'),'--mem='+('32768M'if gpu else'1024M'),'--gres='+('dcu:8'if gpu else'none'),'-w',','.join(NODES),'-t',('00:03:00'if gpu else'00:00:20'),'--job-name='+label,*(['--output='+str(R/'logs'/(label+'-%J.log'))]if gpu else[]),*script]
 def probe(which):
  env=['env','GLM470_EXPECTED_PIDS='+json.dumps(expected,separators=(',',':'))];z=run(command(LABEL+'-'+which,False,env+['python3','-B',str(D/'idle.py')]),25);assert z['rc']==0;rows=[json.loads(v)for v in z['stdout'].splitlines()if v.startswith('{')];assert len(rows)==2 and{v['node']for v in rows}==set(NODES);assert len({(v['node'],pci)for v in rows for pci in v['physical_PCI']})==16;return dict(raw=z,nodes=rows)
 try:
  # Membership and active-step de-dup are fresh, not inherited from CPU/source.
  membership=run(['scontrol','show','job','910371','-o'],12);assert membership['rc']==0;nl=re.search(r'\bNodeList=(\S+)',membership['stdout']);assert nl;hosts=run(['scontrol','show','hostnames',nl[1]],12);assert hosts['rc']==0 and set(NODES)<=set(hosts['stdout'].split());live=run(['squeue','-h','-s','-j','910371','-o','%i|%j|%N'],12);assert live['rc']==0
  for line in live['stdout'].splitlines():
   v=line.split('|');assert len(v)==3
   if re.fullmatch(r'910371[.]\d+',v[0]):
    h=run(['scontrol','show','hostnames',v[2]],8);assert h['rc']==0 and not(set(h['stdout'].split())&set(NODES)),('Live intersecting step',v)
  q['fresh_membership']=membership;q['fresh_active_steps']=live;q['before']=probe('before');assert limit-time.monotonic()>=227,'Insufficient originalkeeper budget beforeGPU';OUT.mkdir(exist_ok=False);epoch=time.time();q['epoch']=epoch;worker_end=time.monotonic()+150;gpu_end=worker_end+30;limit=min(limit,gpu_end+55);args=command(LABEL,True,['env','GLM470_EPOCH='+str(epoch),'GLM470_GPU_MF='+os.environ['GLM470_GPU_MF'],'WAVE_INSIDE_R2323=GLM952-global-TP16-component-r470/device-v2/inside.sh','bash',str(D/'outer.sh'),'probe'])
  with(R/'logs'/(LABEL+'-srun.log')).open('xb')as f:p=subprocess.Popen(args,env=clean_env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  q['submissions'].append(dict(PID=p.pid,PGID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],argv=args));persist()
  try:q['srun_rc']=p.wait(timeout=max(1,gpu_end-time.monotonic()))
  except subprocess.TimeoutExpired:raise TimeoutError('Original180GPU deadline')
  row=account();assert row and row[2:4]==['COMPLETED','0:0']and q['srun_rc']==0;step=row[0];q['GPU']=step;q['GPU_accounting']=row;owners=[]
  for node in range(2):
   w=json.loads((OUT/('worker-%d.json'%node)).read_bytes());assert w['passed']and w['completed']and w['actual_GPU']==step and w['gpu_epoch']==epoch and not w.get('forced_TERM')and not w.get('forced_KILL');assert [v['phase']for v in w['rounds']]==['created','proved'];expected[NODES[node]]=[]
   for own in w['owners']:
    r=json.loads((OUT/('owner-%02d.json'%own['global_rank'])).read_bytes());assert r['completed']and r['passed']and r['normal_close']and r['Runtime_weak_expired_CABI_assertion']and r['actual_GPU']==step and r['gpu_epoch']==epoch and r['global_rank']==node*8+r['device']and r['node']==NODES[node]and r['UID']==os.getuid()and r['owner_TID']==r['PID']==own['PID']and r['startticks']==own['startticks'];assert not r['SDK_first_submit_qualification']and len(r['entry_CPUmask'])>=32 and r['after_create_CPUmask']==r['after_probe_CPUmask']==r['entry_CPUmask'];assert r['proof']['TP_SUM']==136 and r['proof']['TP_order']==list(range(1,17));assert not r['Model_READY']and not r['weights']and not r['cache_allocated']and r['HTTP']==0;assert len(r['native_phase_records'])==6 and [v['phase']for v in r['native_phase_records']]==['allreduce-submit-begin','allreduce-submit-return','allgather-submit-begin','allgather-submit-return','original-sync-begin','original-sync-return']and all(v['rank']==r['global_rank']for v in r['native_phase_records']);expected[NODES[node]].append(dict(PID=r['PID'],startticks=r['startticks'],socket_inodes=r.get('owned_socket_inodes',[])));owners.append(dict(node=r['node'],device=r['device'],global_rank=r['global_rank'],physical_PCI=r['physical_PCI'],PID=r['PID'],report_SHA256=H(OUT/('owner-%02d.json'%r['global_rank']))))
   assert len({r['physical_PCI']for r in owners if r['node']==NODES[node]})==8
  assert json.loads((OUT/'worker-0.json').read_bytes())['rounds'][0]['digest']==json.loads((OUT/'worker-1.json').read_bytes())['rounds'][0]['digest']and json.loads((OUT/'worker-0.json').read_bytes())['rounds'][1]['digest']==json.loads((OUT/'worker-1.json').read_bytes())['rounds'][1]['digest'];assert len(owners)==16 and{r['global_rank']for r in owners}==set(range(16))and len({(r['node'],r['physical_PCI'])for r in owners})==16;assert all(set(r['physical_PCI']for r in owners if r['node']==n['node'])==set(n['physical_PCI'])for n in q['before']['nodes']);q['owners']=owners;q['normal16']=True;q['post']=probe('post');q['passed']=True
 except BaseException as e:
  q.update(error=repr(e),traceback=traceback.format_exc())
  if p is not None and p.poll()is None:
   q['forced_cleanup']=True
   try:
    row=account();step=row[0]if row else step
    if step:
     assert re.fullmatch(r'910371[.]\d+',step);q['owned_step_TERM']=run(['scancel','--signal=TERM',step],8)
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=8)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);q['owned_PGID_KILL']=True;p.wait(timeout=5)
   except BaseException as err:q['cleanup_error']=repr(err)
  for node in range(2):
   f=OUT/('worker-%d.json'%node)
   if f.exists():expected[NODES[node]]=[dict(PID=r['PID'],startticks=r['startticks'])for r in json.loads(f.read_bytes()).get('owners',[])]
  try:q['failure_post']=probe('failure-post')
  except BaseException as err:q['post_error']=repr(err)
 finally:q.update(completed=True,seconds=time.monotonic()-started);persist()
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
