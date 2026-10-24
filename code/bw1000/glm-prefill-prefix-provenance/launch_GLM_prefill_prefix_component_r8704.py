import pathlib,json,hashlib,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();mp=i/'GLM-prefill-prefix-component-manifest-r8704.json';m=json.loads(mp.read_text());go=json.loads((i/'GLM-prefill-prefix-root-GO-r8704.json').read_text());assert go['status']=='CONTINUOUS_FINITE_COMPONENT_GO'and go['DSO_SHA256']==m['candidate_DSO_SHA256'];public=json.loads((i/'GLM-prefill-prefix-component-publication-r8704.json').read_text());assert public['ordinary_push_success']and public['manifest_SHA256']==sha(mp)
for name,h in m['sources_SHA256'].items():assert sha(i/name)==h,name
for path,h in m['DSO_SHA256'].items():assert sha(r/path)==h,path
allocation=subprocess.check_output(['scontrol','show','job','910376','-o'],text=True,timeout=30);nodes=subprocess.check_output(['scontrol','show','hostnames',next(a.split('=',1)[1]for a in allocation.split()if a.startswith('NodeList='))],text=True,timeout=30).splitlines();assert 'a02r4n14'in nodes
intent=r/'GLM-prefill-prefix-component-r8704-intent.json';assert not intent.exists();snapshot=subprocess.check_output(['sacct','-S','2026-10-24','-j','910376','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=30)
busy=set()
for line in snapshot.splitlines():
 f=line.split('|')
 if len(f)>5 and '.'in f[0]and f[0].split('.')[-1].isdigit()and f[2].split()[0].split('+')[0]in['RUNNING','PENDING','COMPLETING','CONFIGURING']:
  busy.update(subprocess.check_output(['scontrol','show','hostnames',f[5]],text=True,timeout=30).splitlines())
selected=None;idle_records=[]
for node in ['a02r4n14','a02r4n15']:
 if node in busy:continue
 idle=subprocess.run(['srun','--jobid=910376','--overlap','--exact','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM-prefill-prefix-idle-r8704','python3',str(i/'verify_gpu_idle_r5819.py')],text=True,capture_output=True,timeout=150)
 rec={'node':node,'returncode':idle.returncode,'stdout':idle.stdout,'stderr':idle.stderr};idle_records.append(rec)
 if idle.returncode==0:
  kd=json.loads(idle.stdout)
  if kd['node']==node and kd['kfd_pids']==[]:selected=node;break
assert selected is not None,('no_idle_node_no_numeric',sorted(busy),idle_records)
args=['srun','--jobid=910376','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c64','-w',selected,'--gres=dcu:8','--mem=128000M','-t','00:10:00','--job-name=GLM-prefill-prefix-component-r8704','--output='+str(r/'logs/GLM-prefill-prefix-component-r8704-%J.log'),'env','WAVE_INSIDE_R2323=GLM_prefill_prefix_component_r8704_inside.sh','bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
v={'state':'unique_intent_before_controller','selected_node':selected,'args':args,'manifest_SHA256':sha(mp),'publication':public,'fresh_idle':kd,'idle_probe_records':idle_records,'fresh_Slurm':snapshot,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with intent.open('x')as f:json.dump(v,f,indent=2)
with(r/'logs/GLM-prefill-prefix-component-r8704-watch-launch.log').open('x')as log:p=subprocess.Popen(['python3',str(i/'controller_GLM_prefill_prefix_component_r8704.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'controller_pid':p.pid,'numeric_args':args,'manifest_SHA256':sha(mp),'one_weight_load':True,'maximum_numeric_attempts':1,'original_watch':str(r/'results/GLM-prefill-prefix-component-r8704-terminal.json')};out=r/'results/GLM-prefill-prefix-component-r8704-launch.json'
with out.open('x')as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
