import pathlib,json,subprocess,os,signal,time,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='kimi-attnres-C9-r8987';p=r/(label+'-launch.json');d=json.loads(p.read_text());receipt=r/'results/Kimi-attnres-C9-pending-cleanup-r8987.json';assert not receipt.exists()
rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910374','-n','-P','-o','JobID,JobName%100,State%40,ExitCode,NodeList%100'],text=True).splitlines()]
matching=[x for x in rows if x[1]==label]
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':label,'CPU_only':True,'no_new_dispatch':True,'original_controller':d,'matching_actual_rows':matching}
if matching:
 out['action']='ACTUAL_HANDLE_FOUND_FOLLOW_ONLY';print(json.dumps(out));raise SystemExit
assert (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(d['submitted_utc'])).total_seconds()>=300
pid=d['pid'];proc=pathlib.Path('/proc')/str(pid)
if proc.exists():
 assert proc.stat().st_uid==os.getuid()
 cmd=(proc/'cmdline').read_bytes().split(b'\0');args=[s.decode()for s in cmd if s]
 assert pathlib.Path(args[0]).name=='srun'and '--jobid=910374'in args and '--job-name='+label in args and '-c64'in args and '--gres=dcu:1'in args and 'a02r4n01'in args,args
 out['verified_srun_args']=args;os.kill(pid,signal.SIGTERM);out['SIGTERM_srun_pid']=pid
 limit=time.monotonic()+15
 while proc.exists()and time.monotonic()<limit:
  state=(proc/'stat').read_text().split()[2]
  if state=='Z':break
  time.sleep(.2)
 assert not proc.exists()or(proc/'stat').read_text().split()[2]=='Z','pending srun did not exit'
wp=d['watcher_pid'];proc=pathlib.Path('/proc')/str(wp)
if proc.exists():
 assert proc.stat().st_uid==os.getuid();args=(proc/'cmdline').read_bytes().split(b'\0');assert any(b'launch_Kimi_attnres_C9_r8987.py'in x for x in args)and b'watch'in args
 os.kill(wp,signal.SIGTERM);out['SIGTERM_CPU_watcher_pid']=wp
z=subprocess.run(['srun','--jobid=910374','--overlap','--exact','-N1','-n1','-c1','-w','a02r4n01','--gres=none','--mem=256M','-t','00:01:00','--job-name='+label+'-post','python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')],capture_output=True,text=True,timeout=60)
out['CPU_post_returncode']=z.returncode;out['CPU_post_stdout']=z.stdout;out['CPU_post_stderr']=z.stderr
if z.returncode==0:
 post=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];assert len(post)==1 and post[0]['passed'];out['post_KFD']=post[0]
out['action']='BOUNDED_PENDING_CANCELLED_NO_NUMERIC';receipt.write_text(json.dumps(out,indent=2));print(json.dumps(out))
