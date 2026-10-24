import datetime,hashlib,json,pathlib,subprocess,sys,time
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='GLM-spawn-trace-r8690'
record=r/'results'/f'{label}-entry.json';state=r/'results'/f'{label}-state.json'
def save(q):
 q['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(state)
if '--execute'in sys.argv:
 q=json.loads(record.read_text());assert q['no_resubmit'];save(q)
 try:
  p=subprocess.run(q['args'],capture_output=True,text=True,timeout=195);q['numeric']={'exit':p.returncode,'stdout':p.stdout[-4000:],'stderr':p.stderr[-4000:]};save(q)
  # Authoritative terminal only: query the unique label, never replay.
  rows=[v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-24','-j','910376','-n','-P','-o','JobID,JobName%80,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=25).splitlines()if v.split('|')[1]==label]
  q['rows']=rows;assert len(rows)==1 and rows[0][2]not in['RUNNING','COMPLETING','PENDING','CONFIGURING'],rows
  post=subprocess.run(['srun','--jobid=910376','--overlap','--exact','-N1','-n1','-c1','-w','a02r4n14','--gres=none','--mem=512M','-t','00:01:00','--job-name='+label+'-post','python3',str(r/'incoming/verify_gpu_idle_r5738.py')],capture_output=True,text=True,timeout=70)
  q['original_post']={'exit':post.returncode,'stdout':post.stdout,'stderr':post.stderr};q['terminal_observed']=True;save(q)
 except BaseException as error:q['UNKNOWN_error']=repr(error);save(q)
 sys.exit(0)
assert not record.exists()and not state.exists(),'originalentryexists_NEVER_RESUBMIT'
m=r/'incoming/GLM-spawn-trace-r8690-manifest.json';manifest=json.loads(m.read_text())
for n,h in manifest['sources_SHA256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
assert hashlib.sha256((r/'containers/dtk26-sandbox-r1/opt/dtk/hip/bin/hipprof').read_bytes()).hexdigest()==manifest['hipprof_exact_ELF_SHA256']
snapshot=subprocess.check_output(['sacct','-S','2026-10-24','-j','910376','-n','-P','-o','JobID,JobName%80,State,ExitCode,NodeList%100'],text=True,timeout=25)
for line in snapshot.splitlines():
 v=line.split('|')
 if '.'in v[0]and v[0].split('.')[-1].isdigit()and v[2]in['RUNNING','PENDING','COMPLETING','CONFIGURING']:
  assert 'a02r4n14'not in subprocess.check_output(['scontrol','show','hostnames',v[4]],text=True,timeout=20).splitlines(),v
idle=subprocess.run(['srun','--jobid=910376','--overlap','--exact','-N1','-n1','-c1','-w','a02r4n14','--gres=none','--mem=512M','-t','00:01:00','--job-name='+label+'-idle','python3',str(r/'incoming/verify_gpu_idle_r5738.py')],capture_output=True,text=True,timeout=70)
assert idle.returncode==0,(idle.stdout,idle.stderr);j=json.loads(next(v for v in idle.stdout.splitlines()if v.startswith('{')));assert j['kfd_pids']==[],j
args=['srun','--jobid=910376','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c16','-w','a02r4n14','--gres=dcu:8','--mem=32768M','-t','00:03:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=GLM_spawn_trace_r8690_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
assert len(sys.argv)==3 and sys.argv[1]=='--published-source-commit'
q={'no_resubmit':True,'maximum_entries':1,'scope':'Tiny8spawn F32sum120/profiletoolvisibility only. NoModel/CCL/weights/nativecompile/performance round.','source_commit':sys.argv[2],'manifest_SHA256':hashlib.sha256(m.read_bytes()).hexdigest(),'fresh_idle':j,'args':args}
with record.open('x')as f:json.dump(q,f,indent=2)
with(r/'logs'/(label+'-controller.log')).open('x')as f:p=subprocess.Popen(['python3',str(r/'incoming/launch_GLM_spawn_trace_r8690.py'),'--execute'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q['controller_pid']=p.pid;save(q);print(json.dumps(q))
