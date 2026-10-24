import pathlib,json,hashlib,subprocess,shutil,shlex,os
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-I192-r8669y';b.mkdir(exist_ok=False);(b/'src').mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
def run(cmd,label):
 d['commands'].append(cmd)
 with(b/(label+'.log')).open('x')as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=190)
 assert z.returncode==0,(label,(b/(label+'.log')).read_text()[-9000:])
try:
 oldp=r/'results/Kimi-CCL512-CPU-r8606.json';assert sha(oldp)=='45e63209189cadc05ed44dd85102f4c1ef505c46f9a5f23ecf099c39515bef52';old=json.loads(oldp.read_text());original=json.loads((r/'results/Kimi-N4-CPU-r8174.json').read_text());proof=json.loads((r/'incoming/Kimi-I192-CPU-source-r8669y.json').read_text())
 for n,h in proof['source_SHA256'].items():assert sha(r/'incoming'/n)==h;shutil.copyfile(r/'incoming'/n,b/'src'/n)
 for p,h in old['all_linked_object_SHA256'].items():assert sha(p)==h,p
 for p,h in old['actual_dynamic_dependency_SHA256'].items():assert sha(p)==h,p
 base=r/'candidates/kimi-joint-expert-5f91ad7-r4620';ops=r/'candidates/rebased-5f91ad7-r3668/Ops/Kimi/InfiniOps';sdk=r/'install/hygon-kimi-5f91ad7-r3672';common=r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM';model=r/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM'
 for n in ['kimi_tiled_w13_r1858.cuh','kimi_reduce_wave_r1044.cuh']:shutil.copyfile(base/n,b/'src'/n)
 provider=b/'provider.o';run(['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-I'+str(b/'src'),'-I'+str(ops/'src'),'-I'+str(ops/'generated'),'-I'+str(sdk/'include'),'-MMD','-MF',str(b/'provider.d'),'-c',str(b/'src/kimi_I192_grouped_provider_r8669y.cu'),'-o',str(provider)],'provider')
 flags=original['commands'][0][:original['commands'][0].index('-MD')];flags.insert(6,'-I'+str(b/'src'));obj=b/'moe.o';run([*flags,'-MD','-MF',str(b/'moe.d'),'-c',str(b/'src/kimi_k3_moe_I192_dispatch_r8669y.cpp'),'-o',str(obj)],'moe')
 target='/refactor/candidates/kimi-N4-tile-CPU-r8174/moe.o';link=list(old['commands'][-1]);assert link.count(target)==1;link[link.index(target)]=str(obj);link.insert(link.index('-o'),str(provider));native=b/'libKimi_I192_r8669y.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(b/'link.map'))if x.startswith('-Wl,-Map,')else x for x in link];run(link,'link')
 inc=['-I/opt/dtk/cuda/cuda/include','-I'+str(model/'csrc'),'-I'+str(common/'csrc'),'-I'+str(sdk/'include'),'-I'+str(common/'third_party/json/include'),'-I'+str(common/'third_party/spdlog/include')];wrapper=b/'libKimi_TP16_I192_r8669y.so'
 run(['g++','-std=c++17','-O2','-fno-omit-frame-pointer','-fPIC','-shared','-D_GLIBCXX_USE_CXX11_ABI=1',*inc,'-MMD','-MF',str(b/'adapter.d'),str(b/'src/kimi_TP16_full_MoE_I192_r8669y.cpp'),'-Wl,--no-as-needed',str(native),str(common/'python/_infinilm.cpython-310-x86_64-linux-gnu.so'),'-Wl,--as-needed','-L'+str(common/'python/infinilm/_lib'),'-linfinilm_runtime','-L'+str(sdk/'lib'),'-linfiniops','-linfinirt','-linfiniccl','-ldl','-Wl,-rpath,'+str(common/'python/infinilm/_lib')+':'+str(sdk/'lib'),'-o',str(wrapper)],'adapter')
 d['fresh_MD']={}
 for md in b.glob('*.d'):
  deps=sorted(set(shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1])));d['fresh_MD'][md.name]={p:sha(p)for p in deps if pathlib.Path(p).is_file()};assert all('/incoming/'not in p for p in deps),md
 needed=subprocess.check_output(['readelf','-d',str(wrapper)],text=True);assert str(native)in needed and old['library']not in needed
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert 'modern_kimi_I192_counts_r8669y'in defs and 'modern_kimi_grouped_I192_r8669y'in defs
 und=subprocess.check_output(['nm','-D','--undefined-only',str(native)],text=True);assert 'modern_kimi_grouped_I192_r8669y'not in und
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),library=str(wrapper),library_SHA256=sha(wrapper),source_SHA256=proof['source_SHA256'],objects={str(p):sha(p)for p in [provider,obj]},unchanged_objects={p:h for p,h in old['all_linked_object_SHA256'].items()if p!=target},Runtime_SHA256=sha(common/'python/infinilm/_lib/libinfinilm_runtime.so'),wrapper_needed=needed,native_defined=defs,native_undefined=und)
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-I192-CPU-r8669y.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native','native_SHA256','library','library_SHA256']}))
