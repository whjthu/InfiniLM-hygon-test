from pathlib import Path
import ast,base64,hashlib,json

D=Path(__file__).resolve().parent;G=D.parents[2];L=G.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=D.parent/'Kimi-long-draft-RPC-CPU-r333';m=json.loads((old/'SOURCE_MANIFEST.json').read_bytes())
deps={'/refactor/incoming/Kimi-long-draft-RPC-CPU-r333/'+n:h for n,h in m['files_SHA256'].items()}
deps['/refactor/incoming/Kimi-long-draft-RPC-CPU-r333/SOURCE_MANIFEST.json']=sha(old/'SOURCE_MANIFEST.json')
cpu=dict(parent=910407,node='a14r2n20',tasks=1,cores=4,host_mem_MiB=8192,total_s=300,inside_s=270,GPU_cards=0,Native_dlopen=False,Model_loads=0,one_attempt=True)
names=['context.py','check_Torch_CPU.py','CPU_main.py','CPU_inside.sh','DESIGN.json','prepare_CPU.py']
for n in names:
 p=D/n
 if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'))
 if p.suffix=='.sh':assert b'\r'not in p.read_bytes()
manifest=dict(files_SHA256={n:sha(D/n)for n in names},dependencies_SHA256=deps,CPU_claim=cpu,GPU_GO=False,scope='Actual reduced40to8 Torch/333Session projection/lifecycle/payload only, no fullwidth projection, no checkpoint/head/Native/GPU')
(D/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8');h=sha(D/'SOURCE_MANIFEST.json')
task=dict(owner='/root/kimi_operator_tuning',state='SOURCE_STATIC_PASS_REAL_TORCH_CPU_PENDING',source=str(D.relative_to(G)).replace('\\','/'),source_manifest_SHA256=h,GPU_GO=False,CPU_claim=cpu,CPU_submitted=False,actual_Torch_CPU='NOT_RUN; fixed localPython has no torch',next='Rootordinarypublish first; one bounded LinuxCPU using existing643/333 environment; no new Slurm/GPU before publication/claim. No Modelreload.',Model_decode_gain='UNKNOWN; originaltotalcommittedappend upper bound near0.105873s/4.912162s=2.155%, rawcat itself smaller; this is primarily prefixretention/TTFT scope, not a main targetdecode claim')
(G/'coordination/tasks/Kimi-raw-history-elision-r384.json').write_text(json.dumps(task,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
payload={n:base64.b64encode((D/n).read_bytes()).decode()for n in names+['SOURCE_MANIFEST.json']}
script="""from pathlib import Path
import ast,base64,hashlib,json,os,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-raw-history-elision-r384';assert not D.exists();D.mkdir()
for n,v in __PAYLOAD__.items():(D/n).write_bytes(base64.b64decode(v))
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==__MF__
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:05:00','--job-name=Kimi-r384-raw-elision-CPU','--output='+str(R/'logs/Kimi-r384-raw-elision-CPU-%J.log'),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
env=dict(os.environ,WAVE_INSIDE_R2323=D.name+'/CPU_inside.sh')
with(R/'logs/Kimi-r384-raw-elision-CPU-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(CPU_submitted=True,GPU=0,Native_dlopen=False,Model_loads=0,PID=p.pid,label='Kimi-r384-raw-elision-CPU',source_manifest_SHA256=__MF__)))
""".replace('__PAYLOAD__',repr(payload)).replace('__MF__',repr(h))
ast.parse(script);(L/'remote384-upload-CPU.py').write_bytes(script.encode('utf8'))
print(json.dumps(dict(source_manifest_SHA256=h,GPU_GO=False,CPU_submitted=False,helper_AST_pass=True)))
