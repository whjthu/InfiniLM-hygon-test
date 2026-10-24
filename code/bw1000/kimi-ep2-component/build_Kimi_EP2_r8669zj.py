import pathlib,json,hashlib,subprocess,shlex,shutil,os
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-EP2-r8669zj';b.mkdir(exist_ok=False);src=b/'src';src.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
def run(cmd,label):
 d['commands'].append(cmd)
 with(b/(label+'.log')).open('x')as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 assert z.returncode==0,(label,(b/(label+'.log')).read_text()[-8000:])
try:
 p=r/'results/Kimi-I192-smoke-static-closure-r8669zg.json';assert sha(p)=='6baffc37e299a5d740644e0b911a1d3b140dd04f445c1b2ee9ea64ab6398007d';old=json.loads(p.read_text());assert old['completed'];assert old['native_SHA256']=='6125640af7f5878e08f00f92527b3c975738bf3fc3563cb2b6afabb5dbbf6fc8'
 proof=json.loads((r/'incoming/Kimi-EP2-source-r8669zj.json').read_text())
 for n,h in proof['source_SHA256'].items():assert sha(r/'incoming'/n)==h;shutil.copyfile(r/'incoming'/n,src/n)
 hp=src/'models/kimi_k3/kimi_k3_moe.hpp';hp.parent.mkdir(parents=True);shutil.copyfile(src/'kimi_k3_moe_EP2_r8669zj.hpp',hp)
 prior=json.loads((r/'results/Kimi-I192-smoke-CPU-r8669zf.json').read_text());cmd=prior['commands'][0];flags=cmd[:cmd.index('-MD')];flags.insert(6,'-I'+str(src));obj=b/'moe.o'
 run([*flags,'-MD','-MF',str(b/'moe.d'),'-c',str(src/'kimi_k3_moe_EP2_r8669zj.cpp'),'-o',str(obj)],'moe')
 sdk=r/'install/hygon-kimi-5f91ad7-r3672';mask=b/'mask.o'
 run(['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-MMD','-MF',str(b/'mask.d'),'-c',str(src/'kimi_EP2_route_mask_r8669zj.cu'),'-o',str(mask)],'mask')
 link=list(prior['commands'][-1]);target='/refactor/candidates/Kimi-I192-smoke-r8669zf/moe.o';assert link.count(target)==1;link[link.index(target)]=str(obj);link.insert(link.index('-o'),str(mask));native=b/'libKimi_EP2_r8669zj.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(b/'link.map'))if v.startswith('-Wl,-Map,')else v for v in link];run(link,'link')
 allobjs={p:sha(p)for p in link if p.endswith('.o')};packed='/refactor/candidates/kimi-packed-reuse-r6136/packed.o';assert allobjs[packed]=='355607f22e0232b6422cef6f81224af24f62b585e5e251fbb656e7eb0da4700c'
 for p,h in old['all_linked_object_SHA256'].items():
  if p!=target:assert allobjs[p]==h,p
 common=r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM';model=r/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM';adapter=b/'libKimi_EP2_full_MoE_r8669zj.so'
 run(['g++','-std=c++17','-O2','-fno-omit-frame-pointer','-fPIC','-shared','-D_GLIBCXX_USE_CXX11_ABI=1','-I'+str(src),'-I/opt/dtk/cuda/cuda/include','-I'+str(model/'csrc'),'-I'+str(common/'csrc'),'-I'+str(sdk/'include'),'-I'+str(common/'third_party/json/include'),'-I'+str(common/'third_party/spdlog/include'),'-MMD','-MF',str(b/'adapter.d'),str(src/'kimi_TP16_EP2_full_MoE_r8669zj.cpp'),'-Wl,--no-as-needed',str(native),str(common/'python/_infinilm.cpython-310-x86_64-linux-gnu.so'),'-Wl,--as-needed','-L'+str(common/'python/infinilm/_lib'),'-linfinilm_runtime','-L'+str(sdk/'lib'),'-linfiniops','-linfinirt','-linfiniccl','-ldl','-Wl,-rpath,'+str(common/'python/infinilm/_lib')+':'+str(sdk/'lib'),'-o',str(adapter)],'adapter')
 d['fresh_MD']={}
 for md in b.glob('*.d'):
  paths=shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]);d['fresh_MD'][md.name]={p:sha(p)for p in paths if pathlib.Path(p).is_file()};assert all('/incoming/'not in p for p in paths)
  if md.name in ['moe.d','adapter.d']:assert str(hp)in d['fresh_MD'][md.name]
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);und=subprocess.check_output(['nm','-D','--undefined-only',str(native)],text=True);assert all(x in defs and x not in und for x in ['modern_kimi_EP2_mask_r8669zj','modern_kimi_EP2_counts_r8669zj','modern_kimi_grouped_r4620','modern_kimi_grouped_I192_r8669y'])
 needed=subprocess.check_output(['readelf','-d',str(adapter)],text=True);assert str(native)in needed and old['native']not in needed
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),library=str(adapter),library_SHA256=sha(adapter),source_SHA256=proof['source_SHA256'],all_linked_object_SHA256=allobjs,objects={str(v):sha(v)for v in [obj,mask]},unchanged_objects={p:h for p,h in allobjs.items()if p not in [str(obj),str(mask)]},wrapper_needed=needed,native_defined=defs,native_undefined=und,scope=proof['scope'])
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-EP2-CPU-r8669zj.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native','native_SHA256','library','library_SHA256']}))
