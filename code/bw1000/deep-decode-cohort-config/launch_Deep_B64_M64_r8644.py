import pathlib,json,hashlib,subprocess,datetime,time,sys
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

r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out=r/'results/Deep-B64-M64-r8644-launch.json';assert not out.exists();manifest_path=r/'incoming/Deep-B64-M64-config-r8644-manifest.json';manifest=json.loads(manifest_path.read_text());assert len(sys.argv)==3 and sys.argv[1]=='--root-approved-manifest-sha256' and sys.argv[2]==sha(manifest_path), 'root_source_review_required_NO_NUMERIC_LAUNCH';expected=manifest['sources_SHA256'];refs={'/refactor/results/deep-attention-r3509-910406.102.json': '91c3d2891266bb9cdaea933bd5eefdd2e239103c1740b6bac475421c391d10c9', '/refactor/results/native-engine-allselected-r816-910373.442.json': '100cce801a5b239a503bd67447a04ada63db34d52c4aa10e08589b71b7c43c41', '/refactor/results/deep-attention-r3511-B8-audit.json': 'bcf82e89560bfd85aea43a113a15cf568e369f5101fd555304b948830c898851', '/refactor/control/deep-chat-routed-r20-910373.162.json': '69ea973b8d20bee5f33e7efbc031fbe2857f8ad844a4a6418969d06450a69ef5', '/refactor/results/modern-deepseek-chat-routed-r20-910373.162-rank0.json': '92633ddb0f61a2a8e5c2c08d25af72f49f30c0d71d6cbaa6d7a3f8d46395e92f'}
for n,h in expected.items():assert sha(r/'incoming'/n)==h,('source_changed',n)
for n,h in refs.items():assert sha(pathlib.Path(n.replace('/refactor',str(r))))==h
lib=r/'candidates/deep-tailSUM-r7212/libdeep_tailSUM_full_r7212.so';assert sha(lib)=='0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea';bp=r/'results/deep-tailSUM-r7212-build.json';assert sha(bp)=='fc7065a9b56255bb710dbedb53a1b71e6c017f7bdc23bc2a86a41664789a3b0b';build=json.loads(bp.read_text())
for n,h in build['link']['full']['objects'].items():assert sha(pathlib.Path(n.replace('/refactor',str(r))))==h
preflight=r/'results/deep-tailSUM-full-r7225-preflight.json';assert sha(preflight)=='a4af40944a9cb78a8788ee3098801b636425974c5a067a188f77bd9d4691db6e';pf=json.loads(preflight.read_text());assert pf['completed']and pf['full_DSO_SHA256']==sha(lib)and pf['build_manifest_SHA256']==sha(bp)
assert pf['completed'] # new Python wrappers pinned independently below
for n,v in pf['actual_DTK_bindings'].items():assert sha(pathlib.Path(n.replace('/refactor',str(r))))==v['SHA256']
for job in ['910374','910377','910381','910384']:
 rows=[v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-24','-j',job,'-n','-P','-o','JobID,State,ExitCode,NodeList%200'],text=True).splitlines()]
 for v in rows:
  if '.'in v[0]and v[0].split('.')[-1].isdigit()and v[1]in['RUNNING','PENDING','CONFIGURING','COMPLETING']:assert 'a02r4n04'not in subprocess.check_output(['scontrol','show','hostnames',v[3]],text=True).splitlines(),('node_busy',v)
probe=subprocess.run(['srun','--jobid=910374','--overlap','--exact','-N1','-n1','-c1','-w','a02r4n04','--gres=none','--mem=512M','-t','00:03:00','--job-name=Deep-B64-M64-idle-r8644','python3',str(r/'incoming/probe_Deep_idle_r4847.py')],capture_output=True,text=True,timeout=165);assert probe.returncode==0,(probe.stdout,probe.stderr);idle=json.loads(next(v for v in probe.stdout.splitlines()if v.startswith('{')));assert idle['passed']and idle['node']=='a02r4n04'
args=['srun','--jobid=910374','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c64','-w','a02r4n04','--gres=dcu:8','--mem=128000M','-t','01:00:00','--job-name=Deep-B64-M64-r8644','--output='+str(r/'logs/Deep-B64-M64-r8644-%J.log'),'env','WAVE_INSIDE_R2323=Deep_B64_M64_r8644_inside.sh','DEEP_USE_RAMCACHE_R3616=0','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
watch_intent=r/'results/deep-B64-M64-r8644-watcher-intent.json'
watch_state=r/'results/deep-B64-M64-r8644-watcher-state.json'
watch_final=r/'results/deep-B64-M64-r8644-watcher.json'
exclusive=r/'results/Deep-B64-M64-r8644-exclusive-intent.json'
for fresh in [watch_intent,watch_state,watch_final,exclusive]:assert not fresh.exists(),('nonfresh_no_numeric_launch',str(fresh))
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launcher_revision':'r8644','args':args,'sources':expected,'references':refs,'DSO_SHA256':sha(lib),'idle':idle,'numeric_spawned':False,'watcher_ready':False}
with exclusive.open('x')as f:json.dump(record,f,indent=2)
try:
 with (r/'logs/Deep-B64-M64-r8644-watch-controller.log').open('x')as watcher_log:
  watcher=subprocess.Popen(['python3',str(r/'incoming/watch_Deep_B64_M64_r8644.py')],stdout=watcher_log,stderr=subprocess.STDOUT,start_new_session=True)
 record['watcher_pid']=watcher.pid
 def spawn_numeric(ready):
  record['readiness']=ready;record['watcher_ready']=True
  tmp=exclusive.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(exclusive)
  assert watcher.poll() is None, 'watcher_dead_after_intent_write_NO_NUMERIC'
  with (r/'logs/Deep-B64-M64-r8644-controller.log').open('x')as log:return subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 p,readiness=start_numeric_after_ready(watcher,watch_intent,watch_state,spawn_numeric)
 record['pid']=p.pid;record['numeric_spawned']=True
except BaseException as failure:
 record['readiness_or_launch_error']=repr(failure)
 tmp=exclusive.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(exclusive)
 with out.open('x')as f:json.dump(record,f,indent=2)
 raise
tmp=exclusive.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(exclusive)
outcome={'launcher_revision':'r8644','readiness':readiness,'numeric_spawned':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'watcher_pid':watcher.pid,'args':args,'sources':expected,'references':refs,'DSO_SHA256':sha(lib),'idle':idle,'maximum_full_weight_loads':1,'precision':5,'layout_HC_tail':[0,0,0],'B_M_I_maxO':[64,[32,64],512,128],'profiling_scope':'plainPython; profile disabled; same fullvocab finite/CPUargmax','no_observer_DSO':True,'process_timeout_seconds':3500,'follow':'samehandle only, noresubmit onobservationtimeout; terminalauditandpostKFD requiredbeforeadoption'};out.write_text(json.dumps(outcome,indent=2));print(json.dumps(outcome))
