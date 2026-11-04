"""Original bounded one-residency keeper; no HTTP, retry or epoch extension."""
from pathlib import Path
import argparse,datetime,json,os,re,socket,subprocess,time,traceback
from support import D,R,PACKAGE,RUN,LABEL,NODE,OUTPUT,TERMINAL,clean_environment,publish,sha,verify
from owned_cleanup import close_owned
from collect import collect,TerminalNotVisible

def gpu_command():
 return ['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','--ntasks-per-node=1','-c32','--gres=dcu:8','--mem=128000M','-w',NODE,'-t','00:45:00','--job-name='+LABEL+'-'+RUN,'--export=ALL','--output='+str(R/'logs'/(LABEL+'-'+RUN+'-%J-%t.log')),'bash',str(D/'resource_wave_host_alias_outer.sh'),'probe']

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--source-only',action='store_true');a=p.parse_args(argv);assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
 m=verify(a.manifest_sha,require_ready=not a.source_only)
 if a.source_only:print(json.dumps(dict(source_verified=True,execution_READY=m['execution_READY'],pending_CPU=m['entry_CPU']is None,Slurm=0,GPU=0,Native_dlopen=False)));return 0
 assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname().split('.')[0]=='a14r2n20'
 keeper_start=time.monotonic();cleanup_limit=keeper_start+100;watch=R/'results'/(LABEL+'-'+RUN+'-watch.json');assert not OUTPUT.exists()and not watch.exists()
 w=dict(completed=False,passed=False,actual_GPU=None,CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],publication_commit=a.publication_commit,deployment_manifest_SHA256=a.manifest_sha,entry_source_manifest_SHA256=m['entry_source_manifest_SHA256'],HTTP=0,planned_requests=5,planned_cards=8,retry=False,reload=False,forced_cleanup=False)
 publish(watch,w);gpu=None;epoch=None;visibility=None
 def save():
  w['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=watch.with_suffix('.tmp');p.write_text(json.dumps(w,indent=2,allow_nan=False)+'\n');p.replace(watch)
 def command(args,timeout=25):return subprocess.check_output(args,text=True,timeout=min(timeout,max(1,cleanup_limit-time.monotonic())))
 def rows():return[x.split('|')for x in command(['sacct','-S','2026-11-04','-j','910374','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100']).splitlines()if'|'in x]
 def hosts(value):return command(['scontrol','show','hostnames',value]).splitlines()
 def exact(label):
  found=[r for r in rows()if r[1]==label];assert len(found)<=1,'Duplicate original profile label'
  if found:assert re.fullmatch(r'910374[.]\d+',found[0][0])and hosts(found[0][4])==[NODE]
  return found[0]if found else None
 def idle(phase):
  known=[];marker=R/'results'/(LABEL+'-'+RUN+'-GPU_PROCESS.json')
  if marker.exists():
   value=json.loads(marker.read_bytes());assert value['node']==NODE and value['actual_step']==w['actual_GPU']and value['gpu_start_epoch']==epoch;assert value['UID']==os.getuid()and type(value['PID'])is int and value['PID']>0 and str(value['startticks']).isdigit();known=[value['PID']]
  env=clean_environment(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',DEEP467_PROFILE_EXPECTED_PIDS=json.dumps({NODE:known}),DEEP467_PROFILE_POST_DEADLINE=str(time.time()+min(90,max(0,cleanup_limit-time.monotonic()))))
  cmd=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=512M','-w',NODE,'-t','00:02:00','--job-name='+LABEL+'-'+RUN+'-'+phase,'python3','-B',str(D/'idle.py')]
  z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=min(115,max(1,cleanup_limit-time.monotonic())));w.setdefault('CPU_idle_observations',[]).append(dict(phase=phase,rc=z.returncode,stdout=z.stdout,stderr=z.stderr));save()
  selected=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')];assert z.returncode==0 and len(selected)==1;q=selected[0];assert q['node']==NODE and q['CPU_only']and not q['GPU_initialized']and q['KFD_available']and not q['kfd_pids']and not q['matching_component_processes']and not q['expected_PIDs_still_exist']and q['ports_bindable']=={'43882':True}and len(set(q['physical_PCI']))==8
  if phase!='before':assert q['physical_PCI']==w['fresh_before']['physical_PCI']
  return q
 try:
  parent=command(['scontrol','show','job','910374','-o']);assert 'JobState=RUNNING'in parent and re.search(r'(?:^| )UserId='+re.escape(os.environ['USER'])+r'\(',parent);assert NODE in hosts(re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1));w['parent_allocation']=parent
  for row in rows():
   assert not row[1].startswith(LABEL+'-'+RUN),'Existing label: reconcile original, no replay'
   if re.fullmatch(r'910374[.]\d+',row[0])and row[2].split()[0].rstrip('+')not in TERMINAL:assert NODE not in hosts(row[4]),'Live overlapping step'
  w['fresh_before']=idle('before');epoch=time.time();mono=time.monotonic();assert mono-keeper_start<=100
  # Global GPU epoch is created once after fresh; cleanup also fits keeper hard3000.
  cleanup_limit=min(keeper_start+2980,mono+2880);w.update(original_GPU_start_unix=epoch,service_deadline_unix=epoch+2400,Native_deadline_unix=epoch+2580,GPU_deadline_unix=epoch+2700,cleanup_deadline_unix=epoch+(cleanup_limit-mono),keeper_hard_deadline_from_entry_s=3000,keeper_startup_elapsed_s=mono-keeper_start);save()
  env=clean_environment(os.environ);env.update(WAVE_INSIDE_R2323='Deep-C1-gap-enclosure-r467/deployment-v1/inside.sh',DEEP467_PROFILE_GPUSTART=str(epoch),DEEP467_PROFILE_DEPLOY_SHA=a.manifest_sha,INFINILM_OWNER_CPU_WIDTH_R2392='4')
  with(R/'logs'/(LABEL+'-'+RUN+'-srun.log')).open('x')as log:gpu=subprocess.Popen(gpu_command(),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  w.update(GPU_srun_PID=gpu.pid,GPU_owned_PGID=gpu.pid,GPU_srun_startticks=Path('/proc/'+str(gpu.pid)+'/stat').read_text().split()[21]);save();terminal_since=None
  while time.time()<epoch+2700:
   try:row=exact(LABEL+'-'+RUN)
   except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as error:
    if gpu.poll()not in(None,0):raise RuntimeError('Owned srun failed during accounting loss')from error
    w.setdefault('transient_accounting_errors',[]).append(repr(error));save();time.sleep(min(15,max(0,epoch+2700-time.time())));continue
   if row:
    assert w['actual_GPU']in(None,row[0]);w.update(actual_GPU=row[0],actual_GPU_Slurm=row)
    if row[2].split()[0].rstrip('+')in TERMINAL:assert row[2:4]==['COMPLETED','0:0'],('Original profile step failed',row)
   assert gpu.poll()in(None,0),'Owned profile srun nonzero'
   if gpu.poll()==0:
    terminal_since=time.monotonic()if terminal_since is None else terminal_since
    if w['actual_GPU']is None:
     assert time.monotonic()-terminal_since<=90,'Original accounting not yet visible'
     w['terminal_awaiting_accounting']=True;save();time.sleep(min(15,max(0,epoch+2700-time.time())));continue
    try:
     result=collect(OUTPUT,w['actual_GPU'],epoch)
     marker=R/'results'/(LABEL+'-'+RUN+'-GPU_PROCESS_CLOSED.json');closed=json.loads(marker.read_bytes())
    except(FileNotFoundError,json.JSONDecodeError,TerminalNotVisible):
     visibility=time.monotonic()if visibility is None else visibility;assert time.monotonic()-visibility<=90,'Original terminal file visibility exceeded90s'
    else:
     if row and row[2:4]==['COMPLETED','0:0']:
      assert closed['normal']and closed['actual_step']==w['actual_GPU']and closed['gpu_start_epoch']==epoch and closed['entry_report_SHA256']==result['entry_report_SHA256'];w.update(terminal=result,normal_GPU_process_exit=True);break
    assert time.monotonic()-terminal_since<=90,'Original accounting/report propagation exceeded90s'
   save();time.sleep(min(15,max(0,epoch+2700-time.time())))
  else:raise TimeoutError('Original GPU2700 epoch expired; no retry or reload')
  w['post']=idle('post');w.update(completed=True,passed=True,state='FIVE_REQUESTS_NORMAL_PARTIAL_EVENT_PROFILE_RECORDED')
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
