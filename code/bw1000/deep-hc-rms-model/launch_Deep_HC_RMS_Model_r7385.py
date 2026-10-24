import pathlib,json,hashlib,subprocess,datetime,time
def wait_watcher_ready(watcher,intent_path,state_path,timeout=30):
 deadline=time.monotonic()+timeout
 while time.monotonic()<deadline:
  assert watcher.poll() is None, 'watcher_exited_before_numeric_launch'
  if intent_path.exists()and state_path.exists():
   try:
    wi=json.loads(intent_path.read_text());ws=json.loads(state_path.read_text())
   except (json.JSONDecodeError,OSError):time.sleep(.2);continue
   assert wi['no_resubmit']and ws['no_resubmit']and ws['resumable_samehandle_only']
   assert ws['step']is None and ws['terminal']is None, 'watcher_not_fresh'
   assert watcher.poll() is None, 'watcher_died_during_readiness'
   return {'watcher_pid':watcher.pid,'same_Popen_alive':True,'fresh_intent_and_state':True,'intent_SHA256':sha(intent_path),'state_SHA256':sha(state_path),'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'numeric_spawned_at_check':False}
  time.sleep(.2)
 raise RuntimeError('watcher_readiness_timeout_NO_NUMERIC_LAUNCH')

def start_numeric_after_ready(watcher,intent_path,state_path,spawn_numeric):
 ready=wait_watcher_ready(watcher,intent_path,state_path)
 assert watcher.poll() is None, 'watcher_died_before_numeric_Popen'
 return spawn_numeric(ready),ready

r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out=r/'results/Deep-HC-RMS-Model-r7385-launch.json';assert not out.exists()
mp=r/'incoming/Deep-HC-RMS-Model-GPU-manifest-r7385.json';manifest=json.loads(mp.read_text())
for name,h in manifest['sources_SHA256'].items():assert sha(r/'incoming'/name)==h,name
lib=r/'candidates/deep-HC-RMS-model-r7383/libdeep_HC_RMS_full_r7383.so';assert sha(lib)==manifest['DSO_SHA256']
for job in ['910374','910377','910381','910384']:
 rows=[v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-23','-j',job,'-n','-P','-o','JobID,State,NodeList%200'],text=True).splitlines()]
 for row in rows:
  if '.'in row[0]and row[0].split('.')[-1].isdigit()and row[1]in['RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED']:assert'a02r4n04'not in subprocess.check_output(['scontrol','show','hostnames',row[2]],text=True).splitlines(),('node_busy',row)
probe=subprocess.run(['srun','--jobid=910374','--overlap','--exact','-N1','-n1','-c1','-w','a02r4n04','--gres=none','--mem=512M','-t','00:03:00','--job-name=Deep-HC-RMS-Model-idle-r7385','python3',str(r/'incoming/probe_Deep_idle_r4847.py')],capture_output=True,text=True,timeout=165);assert probe.returncode==0,(probe.stdout,probe.stderr);idle=json.loads(next(v for v in probe.stdout.splitlines()if v.startswith('{')));assert idle['passed']and idle['node']=='a02r4n04'
args=['srun','--jobid=910374','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c64','-w','a02r4n04','--gres=dcu:8','--mem=128000M','-t','00:45:00','--job-name=Deep-HC-RMS-Model-r7385','--output='+str(r/'logs/Deep-HC-RMS-Model-r7385-%J.log'),'env','WAVE_INSIDE_R2323=Deep_HC_RMS_Model_r7385_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
intent=r/'results/Deep-HC-RMS-Model-r7385-watcher-intent.json';state=r/'results/Deep-HC-RMS-Model-r7385-watcher-state.json';final=r/'results/Deep-HC-RMS-Model-r7385-watcher.json';exclusive=r/'results/Deep-HC-RMS-Model-r7385-exclusive-intent.json'
for p in[intent,state,final,exclusive]:assert not p.exists(),('nonfresh',str(p))
record={'CPU_build':'910374.730','source_manifest_SHA256':sha(mp),'DSO_SHA256':sha(lib),'args':args,'idle':idle,'no_resubmit':True,'numeric_spawned':False}
with exclusive.open('x')as f:json.dump(record,f,indent=2)
try:
 with(r/'logs/Deep-HC-RMS-Model-r7385-watch-controller.log').open('x')as log:watcher=subprocess.Popen(['python3',str(r/'incoming/watch_Deep_HC_RMS_Model_r7385.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record['watcher_pid']=watcher.pid
 def spawn(ready):
  record['readiness']=ready;tmp=exclusive.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(exclusive);assert watcher.poll()is None
  with(r/'logs/Deep-HC-RMS-Model-r7385-controller.log').open('x')as log:return subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 numeric,ready=start_numeric_after_ready(watcher,intent,state,spawn);record.update(pid=numeric.pid,numeric_spawned=True,watcher_ready=True,readiness=ready)
except BaseException as e:
 record['error']=repr(e);out.write_text(json.dumps(record,indent=2));raise
else:
 out.write_text(json.dumps(record,indent=2));tmp=exclusive.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(exclusive);print(json.dumps(record))

