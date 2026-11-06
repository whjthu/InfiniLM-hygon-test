"""One real16owner fullcheckpoint-entry keeper. No automatic rerun/reload."""
from pathlib import Path
import json,os,re,signal,subprocess,time,traceback
from shared import D,R,OUT,LABEL,NODES,H,save,verify,GPU_SECONDS,KEEPER_SECONDS

def clean_env():
 return {k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_')) or k in ['SLURM_CONF','SLURM_CONF_SERVER']}

def main():
 start=time.monotonic();end=start+KEEPER_SECONDS-30;m,cpu,inp,plan=verify(os.environ['GLM470_GPU_MF']);q=dict(completed=False,passed=False,GPU=None,CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],Model_execution=True,Model_qualified=False,quality_qualified=False,HTTP=0,planned=2,keeper_terminal_outcome='UNKNOWN until actual accounting',submissions=[]);watch=R/'results'/(LABEL+'-watch.json');assert not watch.exists() and not OUT.exists();expected={};p=None;step=None;epoch=None
 def persist():save(watch,q)
 def run(args,seconds):
  assert time.monotonic()+seconds<=end,'Original keeper budget';z=subprocess.run(args,env=clean_env(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=seconds);return dict(argv=args,rc=z.returncode,stdout=z.stdout,stderr=z.stderr)
 def account():
  z=run(['sacct','-j','910371','-S','2026-11-05','-n','-P','-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw'],10);assert z['rc']==0;rows=[v.split('|')for v in z['stdout'].splitlines()if v.split('|')[1]==LABEL];assert len(rows)<=1;return rows[0] if rows else None
 def command(label,gpu,script):
  return ['srun','--jobid=910371','--overlap','--exact','--immediate=10','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c'+('64'if gpu else'1'),'--mem='+('224000M'if gpu else'1024M'),'--gres='+('dcu:8'if gpu else'none'),'-w',','.join(NODES),'-t',('04:00:00'if gpu else'00:00:30'),'--job-name='+label,*(['--output='+str(R/'logs'/(label+'-%J.log'))]if gpu else[]),*script]
 def probe(which):
  z=run(command(LABEL+'-'+which,False,['env','GLM470_EXPECTED_PIDS='+json.dumps(expected,separators=(',',':')),'python3','-B',str(D/'idle.py')]),40);assert z['rc']==0;rows=[json.loads(v)for v in z['stdout'].splitlines()if v.startswith('{')];assert len(rows)==2 and {v['node']for v in rows}==set(NODES) and len({(v['node'],pci)for v in rows for pci in v['physical_PCI']})==16;return dict(raw=z,nodes=rows)
 def recover_expected():
  for node in range(2):
   f=OUT/('worker-%d.json'%node)
   if not f.exists():continue
   w=json.loads(f.read_bytes());items=[dict(PID=w['PID'],startticks=w['startticks'],socket_inodes=w.get('owned_socket_inodes',[]))]
   for own in w.get('owners',[]):
    a=OUT/('owner-%02d.json'%own['global_rank']);v=json.loads(a.read_bytes())if a.exists()else own
    items.append(dict(PID=own['PID'],startticks=own['startticks'],socket_inodes=v.get('owned_socket_inodes',[])))
   expected[NODES[node]]=items
 try:
  membership=run(['scontrol','show','job','910371','-o'],12);assert membership['rc']==0;nl=re.search(r'\bNodeList=(\S+)',membership['stdout']);assert nl;h=run(['scontrol','show','hostnames',nl[1]],10);assert h['rc']==0 and set(NODES)<=set(h['stdout'].split());live=run(['squeue','-h','-s','-j','910371','-o','%i|%j|%N'],12);assert live['rc']==0
  for line in live['stdout'].splitlines():
   v=line.split('|');assert len(v)==3
   if re.fullmatch(r'910371[.]\d+',v[0]):
    h=run(['scontrol','show','hostnames',v[2]],8);assert h['rc']==0 and not(set(h['stdout'].split())&set(NODES)),('live intersecting step',v)
  assert time.monotonic()-start<240,'Fresh phase budget exhausted';q['fresh_membership']=membership;q['fresh_active_steps']=live;q['before']=probe('before');assert end-time.monotonic()>GPU_SECONDS+180;OUT.mkdir(exist_ok=False);epoch=time.time();q['epoch']=epoch;gpu_deadline=time.monotonic()+GPU_SECONDS;end=min(end,gpu_deadline+300)
  args=command(LABEL,True,['env','GLM470_EPOCH='+str(epoch),'GLM470_GPU_MF='+os.environ['GLM470_GPU_MF'],'WAVE_INSIDE_R2323=GLM-prefill-memory-r472/full-model-entry-v1/inside.sh','bash',str(D/'outer.sh'),'probe'])
  with(R/'logs'/(LABEL+'-srun.log')).open('xb')as f:p=subprocess.Popen(args,env=clean_env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  q['submissions'].append(dict(PID=p.pid,PGID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],argv=args));persist()
  q['srun_rc']=p.wait(timeout=max(1,gpu_deadline-time.monotonic()))
  # Only bounded terminal visibility reads, never a second launch. Preserve all
  # observations including missing rows; persistent missing is a keeper failure.
  q['terminal_accounting_observations']=[]
  for attempt in range(4):
   row=account();q['terminal_accounting_observations'].append(row);persist()
   if row:break
   if attempt<3:time.sleep(2)
  if row:step=row[0];q['GPU']=step;q['GPU_accounting']=row
  assert row and row[2:4]==['COMPLETED','0:0']and q['srun_rc']==0;owners=[];workers=[]
  for node in range(2):
   w=json.loads((OUT/('worker-%d.json'%node)).read_bytes());assert w['passed']and w['completed']and w['actual_GPU']==step and w['gpu_epoch']==epoch and not w.get('forced_TERM')and not w.get('forced_KILL');assert [v['phase']for v in w['rounds']]==['created','loaded','case0','case1','quiescent','closed'];workers.append(w)
   for dev in range(8):
    r=json.loads((OUT/('owner-%02d.json'%(node*8+dev))).read_bytes());assert r['completed']and r['passed']and r['normal_close']and r['Runtime_weak_expired_CABI_assertion']and r['worker_shutdown_completed'];assert r['global_rank']==node*8+dev and r['node']==NODES[node]and r['device']==dev and r['actual_GPU']==step and r['gpu_epoch']==epoch and r['owner_TID']==r['PID'];assert r['all78layers_loaded']and r['loaded_parameters']==r['checkpoint_key_count']and r['postload_geometry']['Model_READY'];assert len(r['requests'])==2 and r['resets']==4 and r['Native_forward_attempts']==r['Native_forward_completed'];assert r['driver_memory']['min_observed']>=8*1024**3 and not r['driver_memory']['errors'];assert all(v['free']>=8*1024**3 for v in r['memory_boundaries']);assert all(t['completed']and t['reset_before']and t['reset_after']and(t['natural_EOS']or t['cap_hit'])for t in r['requests']);owners.append(r)
   assert len({r['physical_PCI']for r in owners if r['node']==NODES[node]})==8
  assert [v['digest']for v in workers[0]['rounds']]==[v['digest']for v in workers[1]['rounds']];assert all([r['generated_token_ids_SHA256']for r in a['requests']]==[r['generated_token_ids_SHA256']for r in owners[0]['requests']]for a in owners);assert all(set(r['physical_PCI']for r in owners if r['node']==a['node'])==set(a['physical_PCI'])for a in q['before']['nodes']);recover_expected();q['post']=probe('post');q['normal16']=True;q['root_requests']=[{k:r.get(k)for k in ['ordinal','case_id','suite','natural_EOS','cap_hit','visible_nonEOS_nonspecial','backend_decode_tok_s','backend_sampled_TTFT_s','whole_backend_s','score']}for r in owners[0]['requests']];q['passed']=True
 except BaseException as e:
  q.update(error=repr(e),traceback=traceback.format_exc())
  if p is not None and p.poll()is None:
   q['forced_cleanup']=True
   try:
    row=account();step=row[0]if row else step
    if step:
     assert re.fullmatch(r'910371[.]\d+',step);q['owned_step_TERM']=run(['scancel','--signal=TERM',step],8)
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=12)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);q['owned_PGID_KILL']=True;p.wait(timeout=5)
   except BaseException as err:q['cleanup_error']=repr(err)
  try:recover_expected();q['failure_post']=probe('failure-post')
  except BaseException as err:q['post_error']=repr(err)
 finally:q.update(completed=True,seconds=time.monotonic()-start);persist()
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
