import pathlib,json,subprocess,hashlib,os,time,traceback,shlex
r=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;b=r/'candidates/Kimi-W2-pair-r9280';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU_execution':False,'dlopen':False,'new_GPU_TUs':1,'new_HOST_TUs':1,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]}
def run(cmd,label,budget):
 q['commands'].append(cmd);p=subprocess.run(cmd,text=True,capture_output=True,timeout=budget);(b/(label+'.log')).write_bytes((p.stdout+p.stderr).encode());assert p.returncode==0,(label,p.stderr[-12000:])
try:
 current=json.loads((r/'candidates/Kimi-shard-only-Model-r9183/CPU-build.json').read_bytes());assert current['completed'] and sha(current['native'])==current['native_SHA256']=='b4f916dcf32f3bafb7451c94d6a98746a0e6c084edf73aeb6e229275579dbb8d'
 for p,h in current['all_linked_object_SHA256'].items():assert sha(p)==h,p
 old=json.loads((r/'candidates/Kimi-MoE-batch-profile-r9125/CPU-build.json').read_bytes());assert old['completed'];oldprovider=str(r/'candidates/Kimi-MoE-batch-profile-r9125/provider.o');assert sha(oldprovider)==current['all_linked_object_SHA256'][oldprovider]=='f6ae35e870a8730c7f8e564d566503c6c31c1d253ea10758b5a2d3959b2f0c9d'
 original=r/'candidates/Kimi-MoE-batch-profile-r9125/src/kimi_i192_batch_provider_r9125.cu';assert original.read_bytes()==(D/'original_provider_r9125.cu').read_bytes();assert sha(original)==old['source_SHA256'][original.name]
 original_deps=old['dependencies']['provider.d'];assert original_deps[str(original)]==sha(original)
 for p,h in original_deps.items():assert sha(p)==h,('actual provider MD dependency changed',p)
 raw=original.read_text();candidate=(D/'kimi_W2_pair_provider_r9280.cu').read_text()
 # Original W13/routes and original W2 OFF function remain identical source sections.
 sections=[('template<int TT,int Tile>__global__ void groupedW13I192R8669y','template <typename T>'),('template <typename T>\n__global__ void staticW2I192R8669y','\n}\n}')]
 for a,z in sections:
  basepart=raw[raw.index(a):raw.index(z,raw.index(a))];assert basepart in candidate,a
 provider=b/'provider.o';cmd=old['commands'][0][:];cmd=[str(D/'kimi_W2_pair_provider_r9280.cu')if x==str(original)else str(provider)if x==oldprovider else str(b/'provider.d')if x.endswith('/provider.d')else x for x in cmd];run(cmd,'provider',240)
 native=b/'libKimi_W2_pair_r9280.so';cmd=current['commands'][-1][:];assert cmd.count(oldprovider)==1;cmd[cmd.index(oldprovider)]=str(provider);cmd[cmd.index('-o')+1]=str(native)
 for i,x in enumerate(cmd):
  if x.startswith('-Wl,-Map'):cmd[i]='-Wl,-Map,'+str(b/'link.map')
 run(cmd,'native',150);newobjects={p:sha(p)for p in cmd if p.endswith('.o')};unchanged={p:h for p,h in current['all_linked_object_SHA256'].items()if p!=oldprovider};assert all(newobjects[p]==h for p,h in unchanged.items())
 parentRunner=json.loads((r/'candidates/Kimi-current-MoE-profile-r9264/CPU-build.json').read_bytes());assert parentRunner['completed'];cmd=parentRunner['commands'][0][:];cmd=[str(D/'W2_pair_runner_r9280.cpp')if x.endswith('/current_moe_runner_r9264.cpp')else str(native)if x==current['native']else str(b/'adapter.d')if x.endswith('/adapter.d')else str(b/'libKimi_W2_pair_runner_r9280.so')if x==parentRunner['library']else x for x in cmd];run(cmd,'runner',240);library=b/'libKimi_W2_pair_runner_r9280.so'
 env=dict(os.environ,LD_LIBRARY_PATH=parentRunner['LD_LIBRARY_PATH']);p=subprocess.run(['ldd','-r',str(library)],env=env,text=True,capture_output=True,timeout=60);assert p.returncode==0 and'not found'not in p.stdout+p.stderr and'undefined symbol:'not in p.stdout+p.stderr,p.stdout+p.stderr
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);olddefs=subprocess.check_output(['nm','-D','--defined-only',current['native']],text=True);assert'modern_kimi_W2_pair_counts_r9280'in defs;assert all(line.split()[-1]in defs for line in olddefs.splitlines()if len(line.split())>=3),'old exports missing'
 md={}
 for f in b.glob('*.d'):
  paths=shlex.split(f.read_text().replace('\\\n',' ').split(':',1)[1]);md[f.name]={p:sha(p)for p in paths if pathlib.Path(p).is_file()}
 critical='/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3/kimi_k3_moe.hpp';assert md['adapter.d'][critical]==parentRunner['critical_MoE_header_SHA256']
 ldd_text=p.stdout+p.stderr
 deps=subprocess.check_output(['readelf','-d',str(library)],text=True);assert str(native)in deps
 q.update(completed=True,elapsed_s=time.monotonic()-start,native=str(native),native_SHA256=sha(native),library=str(library),library_SHA256=sha(library),original_native_SHA256=current['native_SHA256'],original_provider_SHA256=sha(oldprovider),provider_SHA256=sha(provider),unchanged_objects=unchanged,all_linked_objects=newobjects,original_source_SHA256=sha(original),actual_original_provider_MD=original_deps,all_new_MD=md,critical_MoE_header_SHA256=md['adapter.d'][critical],source_SHA256={p.name:sha(p)for p in D.iterdir()if p.is_file()},LD_LIBRARY_PATH=parentRunner['LD_LIBRARY_PATH'],ldd_relocations=ldd_text,wrapper_DT_NEEDED=deps,defaultOFF=True,one_adjacent_pair_only=True,actual_GPU_not_qualified=True)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-start)
(b/'CPU-build.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','actual_step','native_SHA256','library_SHA256','provider_SHA256','elapsed_s','error']}));assert q['completed']
