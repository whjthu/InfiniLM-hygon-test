"""Original bounded one-residency keeper; no HTTP, retry or epoch extension."""
from pathlib import Path
import argparse,datetime,json,os,re,socket,subprocess,time,traceback
from support import D,R,PACKAGE,RUN,LABEL,NODES,OUTPUT,TERMINAL,clean_environment,publish,sha,verify
from owned_cleanup import close_owned
from collect import collect,TerminalNotVisible

def gpu_command():
 return ['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c64','--gres=dcu:8','--mem=224000M','-w',','.join(NODES),'-t','01:00:00','--job-name='+LABEL+'-'+RUN,'--export=ALL','--output='+str(R/'logs'/(LABEL+'-'+RUN+'-%J-%t.log')),'bash',str(R/'incoming/Deep-C1-gap-enclosure-r467/profile-source-v1/CPU_outer.sh'),'probe']

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--source-only',action='store_true');a=p.parse_args(argv);assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
 m=verify(a.manifest_sha,require_ready=not a.source_only)
 if a.source_only:print(json.dumps(dict(source_verified=True,execution_READY=m['execution_READY'],pending_CPU=False,Slurm=0,GPU=0,Native_dlopen=False)));return 0
 assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname().split('.')[0]=='a14r2n20'
 keeper_start=time.monotonic();cleanup_limit=keeper_start+100;watch=R/'results'/(LABEL+'-'+RUN+'-watch.json');assert not OUTPUT.exists()and not watch.exists()
 w=dict(completed=False,passed=False,actual_GPU=None,CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],publication_commit=a.publication_commit,deployment_manifest_SHA256=a.manifest_sha,entry_source_manifest_SHA256=m['source_manifest_SHA256'],HTTP=0,planned_cases=1,planned_cards=16,retry=False,reload=False,forced_cleanup=False)
 publish(watch,w);gpu=None;epoch=None;visibility=None
 def save():
  w['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=watch.with_suffix('.tmp');p.write_text(json.dumps(w,indent=2,allow_nan=False)+'\n');p.replace(watch)
 def command(args,timeout=25):return subprocess.check_output(args,text=True,timeout=min(timeout,max(1,cleanup_limit-time.monotonic())))
 def rows():return[x.split('|')for x in command(['sacct','-S','2026-11-05','-j','910371','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100']).splitlines()if'|'in x]
 def hosts(value):return command(['scontrol','show','hostnames',value]).splitlines()
 def exact(label):
  found=[r for r in rows()if r[1]==label];assert len(found)<=1,'Duplicate original profile label'
  if found:assert re.fullmatch(r'910371[.]\d+',found[0][0])and hosts(found[0][4])==NODES
  return found[0]if found else None
 def idle(phase):
  env=clean_environment(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',GLM_R425_EXPECTED_PIDS=json.dumps(w.get('terminal',{}).get('expected_node_PIDs',{})),GLM_R425_POST_DEADLINE_UNIX=str(time.time()+min(90,max(0,cleanup_limit-time.monotonic()))))
  cmd=['srun','--jobid=910371','--overlap','--exact','--immediate=20','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','--label','-c1','--gres=none','--mem=512M','-w',','.join(NODES),'-t','00:02:00','--job-name='+LABEL+'-'+RUN+'-'+phase,'python3','-B',str(R/'incoming/GLM-W4-remapped-final-slots-r469/deployment-v1/idle.py')]
  z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=min(115,max(1,cleanup_limit-time.monotonic())));w.setdefault('CPU_idle_observations',[]).append(dict(phase=phase,rc=z.returncode,stdout=z.stdout,stderr=z.stderr));save()
  selected=[json.loads(line.split(': ',1)[1]) for line in z.stdout.splitlines() if re.match(r'^\d+: [{]',line)]
  assert z.returncode==0 and len(selected)==2 and {q['node'] for q in selected}==set(NODES)
  for q in selected:
   assert q['CPU_only'] and not q['GPU_initialized'] and q['KFD_available'] and not q['kfd_pids'] and not q['matching_component_processes'] and not q['expected_PIDs_still_exist'] and all(q['ports_bindable'].values()) and len(set(q['physical_PCI']))==8
   if phase!='before':assert q['physical_PCI']==next(p['physical_PCI'] for p in w['fresh_before'] if p['node']==q['node'])
  return selected

 try:
  parent=command(['scontrol','show','job','910371','-o']);assert 'JobState=RUNNING'in parent and re.search(r'(?:^| )UserId='+re.escape(os.environ['USER'])+r'\(',parent);assert set(NODES)<=set(hosts(re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1)));w['parent_allocation']=parent
  for row in rows():
   assert not row[1].startswith(LABEL+'-'+RUN),'Existing label: reconcile original, no replay'
   if re.fullmatch(r'910371[.]\d+',row[0])and row[2].split()[0].rstrip('+')not in TERMINAL:assert not set(NODES)&set(hosts(row[4])),'Live overlapping step'
  w['fresh_before']=idle('before');epoch=time.time();mono=time.monotonic();assert mono-keeper_start<=100
  # Global GPU epoch is created once after fresh; cleanup also fits keeper hard3900.
  cleanup_limit=min(keeper_start+3880,mono+3780);w.update(original_GPU_start_unix=epoch,service_deadline_unix=epoch+3360,Native_deadline_unix=epoch+3480,GPU_deadline_unix=epoch+3600,cleanup_deadline_unix=epoch+(cleanup_limit-mono),keeper_hard_deadline_from_entry_s=3900,keeper_startup_elapsed_s=mono-keeper_start);save()
  env=clean_environment(os.environ);env.update(WAVE_INSIDE_R2323='GLM-C1-block-logits-verifier/logit-retention-v1/deployment/inside.sh',GLM_PREFIX_EPOCH=str(epoch),GLM_PREFIX_MANIFEST_SHA=a.manifest_sha,INFINILM_OWNER_CPU_WIDTH_R2392='4')
  with(R/'logs'/(LABEL+'-'+RUN+'-srun.log')).open('x')as log:gpu=subprocess.Popen(gpu_command(),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  w.update(GPU_srun_PID=gpu.pid,GPU_owned_PGID=gpu.pid,GPU_srun_startticks=Path('/proc/'+str(gpu.pid)+'/stat').read_text().split()[21]);save();terminal_since=None
  while time.time()<epoch+3600:
   try:row=exact(LABEL+'-'+RUN)
   except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as error:
    if gpu.poll()not in(None,0):raise RuntimeError('Owned srun failed during accounting loss')from error
    w.setdefault('transient_accounting_errors',[]).append(repr(error));save();time.sleep(min(15,max(0,epoch+3600-time.time())));continue
   if row:
    assert w['actual_GPU']in(None,row[0]);w.update(actual_GPU=row[0],actual_GPU_Slurm=row)
    if row[2].split()[0].rstrip('+')in TERMINAL:assert row[2:4]==['COMPLETED','0:0'],('Original profile step failed',row)
   assert gpu.poll()in(None,0),'Owned profile srun nonzero'
   if gpu.poll()==0:
    terminal_since=time.monotonic()if terminal_since is None else terminal_since
    if w['actual_GPU']is None:
     assert time.monotonic()-terminal_since<=90,'Original accounting not yet visible'
     w['terminal_awaiting_accounting']=True;save();time.sleep(min(15,max(0,epoch+3600-time.time())));continue
    try:
     result=collect(OUTPUT,w['actual_GPU'],epoch)
    except(FileNotFoundError,json.JSONDecodeError,TerminalNotVisible):
     visibility=time.monotonic()if visibility is None else visibility;assert time.monotonic()-visibility<=90,'Original terminal file visibility exceeded90s'
    else:
     if row and row[2:4]==['COMPLETED','0:0']:
      w.update(terminal=result,normal_GPU_process_exit=True);break
    assert time.monotonic()-terminal_since<=90,'Original accounting/report propagation exceeded90s'
   save();time.sleep(min(15,max(0,epoch+3600-time.time())))
  else:raise TimeoutError('Original GPU3600 epoch expired; no retry or reload')
  w['post']=idle('post');w.update(completed=True,passed=True,state='ONE_CASE_LOGIT_RETENTION_NORMAL16')
 except BaseException as error:
  w.update(completed=True,passed=False,state='FAILED_OR_UNKNOWN_NO_RETRY',error=repr(error),traceback=traceback.format_exc())
  if gpu is not None:
   w['forced_cleanup']=True
   try:w['owned_cleanup']=close_owned(LABEL+'-'+RUN,gpu,w['actual_GPU'],exact,cleanup_limit)
   except BaseException as cleanup:w['cleanup_error']=repr(cleanup)
  if 'fresh_before'in w:
   try:w['post_after_failure']=idle('failure-post')
   except BaseException as cleanup:w['post_error']=repr(cleanup)
 finally:save()
 print(json.dumps({k:w.get(k)for k in['completed','passed','actual_GPU','CPU_keeper','state','error']}));return 0 if w['passed']else 1
if __name__=='__main__':raise SystemExit(main())
