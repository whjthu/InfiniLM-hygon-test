import sys
assert len(sys.argv)==2,'RootreviewedmanifestSHArequired_NO_GPU'
ROOT_REVIEWED_MANIFEST_SHA256=sys.argv[1]
import pathlib,json,subprocess,datetime,hashlib,uuid,time
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out=r/'glm-transient-Model-r8864-launch.json';assert not out.exists()
m=r/'incoming/GLM-CROSSREQUEST-Model-source-manifest-r8864.json';assert hashlib.sha256(m.read_bytes()).hexdigest()==ROOT_REVIEWED_MANIFEST_SHA256;manifest=json.loads(m.read_text())
for name,h in manifest['source_SHA256'].items():assert hashlib.sha256((r/'incoming'/name).read_bytes()).hexdigest()==h,('source',name)
for name,h in manifest['DSO_SHA256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,('DSO',name)
allocation=subprocess.check_output(['scontrol','show','job','910371','-o'],text=True,timeout=30);assert 'JobState=RUNNING'in allocation,allocation
allocated_nodes=subprocess.check_output(['scontrol','show','hostnames',next(v.split('=',1)[1]for v in allocation.split()if v.startswith('NodeList='))],text=True,timeout=30).splitlines();assert set(manifest['DP_nodes_by_slot'])<=set(allocated_nodes)
nodes=['a02r3n03', 'a02r3n07', 'a02r3n15', 'a02r3n16'];snapshot=subprocess.check_output(['sacct','-S','2026-10-23','-j','910371','-n','-P','-o','JobID,JobName%100,State,NodeList%100'],text=True,timeout=30)
for line in snapshot.splitlines():
 v=line.split('|')
 if '.'in v[0]and v[0].split('.')[-1].isdigit()and v[2]in ['RUNNING','PENDING','CONFIGURING','COMPLETING']:
  hosts=subprocess.check_output(['scontrol','show','hostnames',v[3]],text=True,timeout=25).splitlines();assert not set(nodes)&set(hosts),v
p=subprocess.run(['srun','--jobid=910371','--overlap','--exact','-N4','-n4','--ntasks-per-node=1','-c1','-w',','.join(nodes),'--gres=none','--mem=1000M','-t','00:02:00','--job-name=glm-transient-Model-idle-r8864','python3',str(r/'incoming/verify_gpu_idle_r5738.py')],text=True,capture_output=True,timeout=90);assert p.returncode==0,(p.stdout,p.stderr);idle=[json.loads(x)for x in p.stdout.splitlines()if x.strip()];assert sorted(x['node']for x in idle)==nodes and all(x['kfd_pids']==[]for x in idle)
run='DP32-8864-'+uuid.uuid4().hex[:20]
args=['srun', '--jobid=910371', '--overlap', '--exact', '--kill-on-bad-exit=1', '-N4', '-n4', '--ntasks-per-node=1', '-c64', '-w', 'a02r3n03,a02r3n07,a02r3n15,a02r3n16', '--gres=dcu:8', '--mem=240000M', '-t', '01:00:00', '--job-name=glm-transient-Model-r8864', '--output=/public/home/aclsylqidf/hygon-inference-refactor-20261014/logs/glm-transient-Model-r8864-%J-%t.log', 'env', 'GLM_SHARED_PREFIX_VALIDATE_R5814=0', 'GLM_PREFLIGHT_ONLY_R5194=0', 'GLM_DECODE_ROW_DEDUP_R5386=0', 'GLM_RUN_R5040=REPLACE_RUN', 'GLM_BOUNDED_ROPE_R4279=1', 'GLM_SHARED_CONTROL_R4279=1', 'GLM_PP_SIZE_R3603=2', 'GLM_CACHE_ROWS_R3627=32', 'GLM_CACHE_CAP_R3627=640', 'GLM_NETWORK_TRIAL=shca', 'GLM_PIPELINE_TAIL_NODE=a02r3n07', 'GLM_EXACT_BANKS_R4188=1', 'GLM_CHANNELS_R4363=0', 'GLM_VARIANT_R4363=layer77-5783', 'INFINILM_OWNER_CPU_WIDTH_R2392=4', 'WAVE_INSIDE_R2323=glm_prefill_micro_Model_r8864_inside.sh', 'USE_WEIGHT_CACHE_R3604=0', 'bash', '/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/resource_wave_local_r3097_outer.sh', 'probe']
args=[a.replace('REPLACE_RUN',run)for a in args]
with out.open('x')as f:json.dump({'state':'launch_intent_no_numeric','run':run,'args':args,'source_manifest_SHA256':hashlib.sha256(m.read_bytes()).hexdigest()},f,indent=2)
# Observer must be alive and publishing the exactrun BEFORE numericPopen.
side=r/'incoming/GLM-transient-Model-observer-manifest-r8864.json';assert hashlib.sha256(side.read_bytes()).hexdigest()==json.loads(m.read_text())['observer_manifest_SHA256']
for n,h in json.loads(side.read_text())['source_SHA256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
watchintent=r/'GLM-transient-Model-watch-r8864-launch.json'
with watchintent.open('x')as f:json.dump({'state':'unique_before_watch_Popen','run':run,'Model_submission':False},f)
with(r/'logs/GLM-transient-Model-watch-r8864.log').open('x')as f:watch=subprocess.Popen(['python3',str(r/'incoming/watch_GLM_prefill_Model_terminal_r8864.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
watchstate=r/'results/GLM-transient-Model-watch-r8864.json';ready=None;deadline=time.monotonic()+90
while time.monotonic()<deadline and watch.poll()is None:
 if watchstate.exists():
  state=json.loads(watchstate.read_text())
  if state.get('state')=='watching'and state.get('run')==run and state.get('observer_manifest_SHA256')==hashlib.sha256(side.read_bytes()).hexdigest():ready=state;break
 time.sleep(.5)
if ready is None or watch.poll()is not None:
 out.write_text(json.dumps({'state':'watch_not_ready_NO_NUMERIC','run':run,'watch_pid':watch.pid,'watch_returncode':watch.poll(),'numeric_started':False},indent=2));raise RuntimeError('watch not ready; no numeric Popen')
with(r/'logs/glm-transient-Model-r8864-controller.log').open('x')as f:controller=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':run,'pid':controller.pid,'watch_pid':watch.pid,'watch_ready_same_run':ready,'args':args,'source_manifest_SHA256':hashlib.sha256(m.read_bytes()).hexdigest(),'fresh_KFD_idle':idle,'fresh_sacct':snapshot,'scope':'Two independent world16 TP8PP2 B16M8 replicas A/B split32distinct477. One load perreplica, isoA/isoB/joint, each2full128arms; commonCPUcollector and localemitter timing separate;32normal/audit/finallypost4nodes;60minhard3500inner. DP throughput UNKNOWN.'};out.write_text(json.dumps(d,indent=2));print(json.dumps(d))
