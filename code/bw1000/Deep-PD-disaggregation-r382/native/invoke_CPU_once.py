"""Remote host launcher; requires published immutable source and CPU-only claim."""
import hashlib,json,os,pathlib,subprocess
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/Deep-PD-disaggregation-r382/native'
mf=D/'SOURCE_MANIFEST.json'
manifest=json.loads(mf.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(D/n)==h for n,h in manifest['files_SHA256'].items())
assert not (R/'candidates/Deep-PD-native-r382-cpu-v1').exists()
intent=R/'results/Deep382-native-CPU-v1-once-intent.json'
with intent.open('x') as f:
    json.dump({'manifest_SHA256':sha(mf),'CPU_only':True,'GPU':0,'parent':910407,'node':'a14r2n20','cores':16,'mem_MiB':32768,'limit_s':900},f);f.flush();os.fsync(f.fileno())
env=os.environ.copy()
env['DEEP_PD_NATIVE_MANIFEST_SHA256']=sha(mf)
env['WAVE_INSIDE_R2323']='Deep-PD-disaggregation-r382/native/CPU_inside.sh'
argv=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c16','-w','a14r2n20','--gres=none','--mem=32768M','-t','00:15:00','--job-name=Deep382-native-HOST-CPU-v1','--export=ALL','bash',str(R/'incoming/Deep-C1-current-profile-r353/repair-r1/controller-v2/resource_wave_host_alias_outer.sh'),'probe']
log=R/'logs/Deep382-native-CPU-v1-srun.log'
with log.open('x') as f:
    proc=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_PID':proc.pid,'argv':argv,'log':str(log),'manifest_SHA256':sha(mf),'GPU':0,'Native_dlopen':False}))
