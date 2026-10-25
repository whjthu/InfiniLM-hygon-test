import pathlib,json,hashlib,subprocess,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming';tag='Kimi-projection-real-r9030';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((i/'Kimi-projection-manifest-r9030.json').read_text());public=json.loads((i/'Kimi-projection-rootGO-r9030.json').read_text());assert public['root_continuous_GO']
for n,h in m['sources_SHA256'].items():assert sha(i/n)==h,n
assert sha(r/m['native'].removeprefix('/refactor/'))==m['native_SHA256'];assert sha(i/'verify_gpu_idle_r5819.py')==m['idle_helper_SHA256']
node='a02r4n04';allocation=subprocess.check_output(['scontrol','show','job','910374','-o'],text=True,timeout=20);assert 'JobState=RUNNING'in allocation
nodes=subprocess.check_output(['scontrol','show','hostnames',next(a.split('=',1)[1]for a in allocation.split()if a.startswith('NodeList='))],text=True,timeout=20).splitlines();assert node in nodes
snapshot=subprocess.check_output(['sacct','-S','2026-10-25','-j','910374','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True,timeout=20)
for line in snapshot.splitlines():
 f=line.split('|')
 if len(f)>5 and '.'in f[0]and f[0].split('.')[-1].isdigit()and f[2].split()[0].rstrip('+')in ['RUNNING','PENDING','COMPLETING','CONFIGURING']:
  assert node not in subprocess.check_output(['scontrol','show','hostnames',f[5]],text=True,timeout=20).splitlines(),('node busy',f)
intent=r/(tag+'-intent.json');assert not intent.exists()
idle=subprocess.run(['srun','--jobid=910374','--overlap','--exact','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name=Kimi-projection-idle-r9030','python3',str(i/'verify_gpu_idle_r5819.py')],text=True,capture_output=True,timeout=150);assert idle.returncode==0,(idle.stdout,idle.stderr);kd=json.loads(idle.stdout);assert kd['node']==node and kd['kfd_pids']==[]
wave='Kimi_projection_GPU_r9030_inside.sh';assert wave==m['WAVE_INSIDE']and sha(i/wave)==m['sources_SHA256'][wave]
args=['srun','--jobid=910374','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c8','-w',node,'--gres=dcu:1','--mem=32768M','-t','00:04:00','--job-name='+tag,'--output='+str(r/'logs'/(tag+'-%J.log')),'env','WAVE_INSIDE_R2323='+wave,'bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
v={'selected_node':node,'args':args,'source_commit':public['commit'],'fresh_idle':kd,'manifest_SHA256':sha(i/'Kimi-projection-manifest-r9030.json'),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with intent.open('x')as f:json.dump(v,f,indent=2)
with(r/'logs'/(tag+'-watch-launch.log')).open('x')as f:p=subprocess.Popen(['python3',str(i/'controller_Kimi_projection_r9030.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'controller_pid':p.pid,'watch':str(r/'results'/(tag+'-terminal.json')),'intent':v,'numeric_attempts_max':1,'post_attempts_max':1}))
