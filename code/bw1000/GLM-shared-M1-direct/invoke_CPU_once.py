import pathlib,hashlib,json,subprocess,os
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-shared-M1-direct'
mf=D/'SOURCE_MANIFEST.json';m=json.loads(mf.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(D/n)==h for n,h in m['files_SHA256'].items());assert not(R/'candidates/GLM-shared-M1-direct-cpu-v1').exists()
intent=R/'results/GLM-shared-M1-direct-CPU-once-intent.json'
with intent.open('x')as f:json.dump({'source_manifest_SHA256':sha(mf),'CPU_only':True,'GPU':0},f);f.flush();os.fsync(f.fileno())
env=os.environ.copy();env['GLM_SHARED_M1_SOURCE_SHA256']=sha(mf);env['WAVE_INSIDE_R2323']='GLM-shared-M1-direct/CPU_inside.sh'
argv=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:05:00','--job-name=GLM-shared-M1-HOST-CPU-v1','--export=ALL','bash',str(R/'incoming/Deep-C1-current-profile-r353/repair-r1/controller-v2/resource_wave_host_alias_outer.sh'),'probe']
log=R/'logs/GLM-shared-M1-direct-CPU-v1.log'
with log.open('x')as f:p=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_PID':p.pid,'argv':argv,'log':str(log),'source_manifest_SHA256':sha(mf),'GPU':0}))
