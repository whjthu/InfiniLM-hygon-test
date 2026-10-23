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

r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out=r/'results/Deep-causal-teacher-r7238-launch.json';assert not out.exists();expected={'modern_Deep_causal_teacher_r7238.py': 'd53b8abff43cb3767f166a2a0b6b09ffce84244ee46474e127d653f4d7fe691c', 'Deep_causal_teacher_matrix_r7238.py': 'e6a49bd259d3267350843481a3aaa85e54560b8d23a5ed6ffde24525cf3425e4', 'Deep_causal_teacher_r7238_inside.sh': 'd8a062f881ea9a828beaa4d28894b66c754334a8c80ef0387961cb5015f5a0b4', 'deep_causal_observation_r7233.py': '42f8c1cc7aebf55475bfe018bd95621f7f4f2fad71865e38dde32488216ee79b', 'audit_Deep_causal_teacher_r7238.py': '32717d29d1e7fa17fbfcc080aa88e88891c81130838215cacfc48eed13691758', 'watch_Deep_causal_teacher_r7238.py': '2a9ea5eb770a53adb81d06a86e9d1a09d85785b04a7d51f4b2d6af9287e66c66', 'model_affinity_check_r2628.py': 'c439554f0e942078abdf880f3e872959389715d3650f2b894881fe942e8a36f4', 'deep_affinity_env_contract_r7166.py': '4e8695a543439a01a4dbebf7eab285f0808946dd26ad31935feb638e731a4f33', 'build_deep_serving_cases_r2130.py': '122afbf4f4d6778757bbd3f91ae0e930bb85f826e282b272351a6689ac5491d9', 'probe_Deep_idle_r4847.py': '75542ce3dde6e40ed7ab6a1e5dd8ba69439a7e6fa8fb13011b65ef6b1e77b7eb', 'resource_wave_local_r3097_outer.sh': 'd3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'};refs={'/refactor/results/deep-attention-r3509-910406.102.json': '91c3d2891266bb9cdaea933bd5eefdd2e239103c1740b6bac475421c391d10c9', '/refactor/results/native-engine-allselected-r816-910373.442.json': '100cce801a5b239a503bd67447a04ada63db34d52c4aa10e08589b71b7c43c41', '/refactor/results/deep-attention-r3511-B8-audit.json': 'bcf82e89560bfd85aea43a113a15cf568e369f5101fd555304b948830c898851', '/refactor/control/deep-chat-routed-r20-910373.162.json': '69ea973b8d20bee5f33e7efbc031fbe2857f8ad844a4a6418969d06450a69ef5', '/refactor/results/modern-deepseek-chat-routed-r20-910373.162-rank0.json': '92633ddb0f61a2a8e5c2c08d25af72f49f30c0d71d6cbaa6d7a3f8d46395e92f'}
for n,h in expected.items():assert sha(r/'incoming'/n)==h,('source_changed',n)
for n,h in refs.items():assert sha(pathlib.Path(n.replace('/refactor',str(r))))==h
lib=r/'candidates/deep-tailSUM-r7212/libdeep_tailSUM_full_r7212.so';assert sha(lib)=='0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea';bp=r/'results/deep-tailSUM-r7212-build.json';assert sha(bp)=='fc7065a9b56255bb710dbedb53a1b71e6c017f7bdc23bc2a86a41664789a3b0b';build=json.loads(bp.read_text())
for n,h in build['link']['full']['objects'].items():assert sha(pathlib.Path(n.replace('/refactor',str(r))))==h
preflight=r/'results/deep-tailSUM-full-r7225-preflight.json';assert sha(preflight)=='a4af40944a9cb78a8788ee3098801b636425974c5a067a188f77bd9d4691db6e';pf=json.loads(preflight.read_text());assert pf['completed']and pf['full_DSO_SHA256']==sha(lib)and pf['build_manifest_SHA256']==sha(bp)
assert pf['completed'] # new Python wrappers pinned independently below
for n,v in pf['actual_DTK_bindings'].items():assert sha(pathlib.Path(n.replace('/refactor',str(r))))==v['SHA256']
for job in ['910374','910377','910381','910384']:
 rows=[v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-23','-j',job,'-n','-P','-o','JobID,State,ExitCode,NodeList%200'],text=True).splitlines()]
 for v in rows:
  if '.'in v[0]and v[0].split('.')[-1].isdigit()and v[1]in['RUNNING','PENDING','CONFIGURING','COMPLETING']:assert 'a02r4n05'not in subprocess.check_output(['scontrol','show','hostnames',v[3]],text=True).splitlines(),('node_busy',v)
probe=subprocess.run(['srun','--jobid=910374','--overlap','--exact','-N1','-n1','-c1','-w','a02r4n05','--gres=none','--mem=512M','-t','00:03:00','--job-name=Deep-causal-teacher-idle-r7238','python3',str(r/'incoming/probe_Deep_idle_r4847.py')],capture_output=True,text=True,timeout=165);assert probe.returncode==0,(probe.stdout,probe.stderr);idle=json.loads(next(v for v in probe.stdout.splitlines()if v.startswith('{')));assert idle['passed']and idle['node']=='a02r4n05'
args=['srun','--jobid=910374','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c64','-w','a02r4n05','--gres=dcu:8','--mem=128000M','-t','00:40:00','--job-name=Deep-causal-teacher-r7238','--output='+str(r/'logs/Deep-causal-teacher-r7238-%J.log'),'env','WAVE_INSIDE_R2323=Deep_causal_teacher_r7238_inside.sh','DEEP_USE_RAMCACHE_R3616=0','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
watch_intent=r/'results/deep-causal-teacher-r7238-watcher-intent.json'
watch_state=r/'results/deep-causal-teacher-r7238-watcher-state.json'
watch_final=r/'results/deep-causal-teacher-r7238-watcher.json'
exclusive=r/'results/Deep-causal-teacher-r7238-exclusive-intent.json'
for fresh in [watch_intent,watch_state,watch_final,exclusive]:assert not fresh.exists(),('nonfresh_no_numeric_launch',str(fresh))
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launcher_revision':'r7238','args':args,'sources':expected,'references':refs,'DSO_SHA256':sha(lib),'idle':idle,'numeric_spawned':False,'watcher_ready':False}
with exclusive.open('x')as f:json.dump(record,f,indent=2)
try:
 with (r/'logs/Deep-causal-teacher-r7238-watch-controller.log').open('x')as watcher_log:
  watcher=subprocess.Popen(['python3',str(r/'incoming/watch_Deep_causal_teacher_r7238.py')],stdout=watcher_log,stderr=subprocess.STDOUT,start_new_session=True)
 record['watcher_pid']=watcher.pid
 def spawn_numeric(ready):
  record['readiness']=ready;record['watcher_ready']=True
  tmp=exclusive.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(exclusive)
  assert watcher.poll() is None, 'watcher_dead_after_intent_write_NO_NUMERIC'
  with (r/'logs/Deep-causal-teacher-r7238-controller.log').open('x')as log:return subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 p,readiness=start_numeric_after_ready(watcher,watch_intent,watch_state,spawn_numeric)
 record['pid']=p.pid;record['numeric_spawned']=True
except BaseException as failure:
 record['readiness_or_launch_error']=repr(failure)
 tmp=exclusive.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(exclusive)
 with out.open('x')as f:json.dump(record,f,indent=2)
 raise
tmp=exclusive.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(exclusive)
outcome={'launcher_revision':'r7238','readiness':readiness,'numeric_spawned':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'watcher_pid':watcher.pid,'args':args,'sources':expected,'references':refs,'DSO_SHA256':sha(lib),'idle':idle,'maximum_full_weight_loads':1,'precision':5,'layout_HC_tail':[[0,0,0]],'decode_projection_C4':[[16,16],[0,16],[16,0]],'no_observer_DSO':True,'process_timeout_seconds':2300,'follow':'samehandle only, noresubmit onobservationtimeout; terminalauditandpostKFD requiredbeforeadoption'};out.write_text(json.dumps(outcome,indent=2));print(json.dumps(outcome))
