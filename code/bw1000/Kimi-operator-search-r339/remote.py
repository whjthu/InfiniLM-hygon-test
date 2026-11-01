import pathlib,subprocess,json,sys,base64,hashlib
D=pathlib.Path(__file__).resolve().parent;L=D.parents[3];R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
def remote(code):
 args=['ssh','-T','-p','65032','-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(L/'known_hosts'),'aclsylqidf@zzeshell.scnet.cn','python3 -']
 p=subprocess.run(args,input=code,capture_output=True,text=True,timeout=190)
 assert p.returncode==0,(p.stdout,p.stderr)
 return json.loads(p.stdout)
if sys.argv[1]=='transfer':
 payload={p.name:base64.b64encode(p.read_bytes()).decode() for p in D.iterdir() if p.is_file()}
 code="""from pathlib import Path
import json,base64,hashlib
D=Path(%r)/'incoming/Kimi-operator-search-r339';D.mkdir(exist_ok=True)
payload=%r
for n,v in payload.items():
 raw=base64.b64decode(v)
 if (D/n).exists():assert (D/n).read_bytes()==raw,n
 else:(D/n).write_bytes(raw)
print(json.dumps({'transferred':len(payload),'manifest_SHA256':hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()}))
"""%(R,payload)
elif sys.argv[1]=='cpu':
 code="""import subprocess,json,pathlib
R=pathlib.Path(%r);D=R/'incoming/Kimi-operator-search-r339'
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:10:00','--job-name=Kimi339-CPU','--output='+str(D/'CPU-job.log'),'env','WAVE_INSIDE_R2323=Kimi-operator-search-r339/CPU_inside.sh','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
p=subprocess.Popen(args,stdout=(D/'CPU-submit.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_submit_pid':p.pid,'args':args}))
"""%R
elif sys.argv[1]=='read':
 code="""from pathlib import Path
import json,subprocess
D=Path(%r)/'incoming/Kimi-operator-search-r339'
print(json.dumps({'logs':{n:(D/n).read_text()[-16000:] for n in ['CPU-submit.log','CPU-job.log','compile.log'] if (D/n).exists()},'binary_exists':(D/'search.so').exists(),'accounting':subprocess.check_output(['sacct','-j','910407','-S','2026-11-02','-n','-P','-o','JobID,JobName%%50,State,ExitCode,NodeList'],text=True)[-14000:]}))
"""%R
else:raise ValueError(sys.argv[1])
print(json.dumps(remote(code),ensure_ascii=False,indent=2))
