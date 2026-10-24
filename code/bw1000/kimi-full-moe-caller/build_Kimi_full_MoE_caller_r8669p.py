import pathlib,json,hashlib,subprocess,shutil,os,shlex
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-full-MoE-caller-r8669p';b.mkdir(exist_ok=False);c=r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM';m=r/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM';sdk=r/'install/hygon-kimi-5f91ad7-r3672';native=r/'candidates/Kimi-CCL512-r8606/libKimi_CCL512_r8606.so';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();d={'CPU_only':True,'GPU':False,'dlopen':False,'completed':False,'commands':[]}
def run(cmd):
 d['commands'].append(cmd);z=subprocess.run(cmd,capture_output=True,text=True,timeout=180);assert z.returncode==0,z.stderr[-10000:];return z.stdout
try:
 assert sha(native)=='f6f10c2b243e0336074b872859e6ca6acb08effdcd56d86b9ac4df1bf2ddb5e9'
 for n in ['kimi_full_moe_caller_r8669p.cpp','kimi_HIP_caller_sample_r8669o.cpp']:shutil.copyfile(r/'incoming'/n,b/n)
 inc=['-I/opt/dtk/cuda/cuda/include','-I'+str(m/'csrc'),'-I'+str(c/'csrc'),'-I'+str(sdk/'include'),'-I'+str(c/'third_party/json/include'),'-I'+str(c/'third_party/spdlog/include')]
 common=['g++','-std=c++17','-O2','-fno-omit-frame-pointer','-fPIC','-shared','-D_GLIBCXX_USE_CXX11_ABI=1',*inc]
 lib=b/'libKimi_full_MoE_caller_r8669p.so';sample=b/'libKimi_HIP_caller_sample_r8669p.so'
 run([*common,'-MMD','-MF',str(b/'adapter.d'),str(b/'kimi_full_moe_caller_r8669p.cpp'),'-Wl,--no-as-needed',str(native),str(c/'python/_infinilm.cpython-310-x86_64-linux-gnu.so'),'-Wl,--as-needed','-L'+str(c/'python/infinilm/_lib'),'-linfinilm_runtime','-L'+str(sdk/'lib'),'-linfiniops','-linfinirt','-linfiniccl','-ldl','-Wl,-rpath,'+str(c/'python/infinilm/_lib')+':'+str(sdk/'lib'),'-o',str(lib)])
 run([*common,'-MMD','-MF',str(b/'sampler.d'),str(b/'kimi_HIP_caller_sample_r8669o.cpp'),'-ldl','-o',str(sample)])
 d['fresh_MD']={}
 for md in [b/'adapter.d',b/'sampler.d']:
  paths=shlex.split(md.read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]);d['fresh_MD'][md.name]={p:sha(pathlib.Path(p))for p in paths if pathlib.Path(p).is_file()}
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],library=str(lib),library_SHA256=sha(lib),sampler=str(sample),sampler_SHA256=sha(sample),source_SHA256={p.name:sha(p)for p in b.glob('*.cpp')},native_SHA256=sha(native),Runtime_SHA256=sha(c/'python/infinilm/_lib/libinfinilm_runtime.so'),needed=run(['readelf','-d',str(lib)]),sampler_needed=run(['readelf','-d',str(sample)]),undefined=run(['nm','-D','-C','--undefined-only',str(lib)]),native_defined=run(['nm','-D','-C','--defined-only',str(native)]))
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-full-MoE-caller-CPU-r8669p.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','library','library_SHA256','sampler_SHA256']}))
