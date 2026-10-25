import pathlib,json,hashlib,subprocess,shutil,shlex,os
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-official-WFP4-r8669zu';b.mkdir(exist_ok=False);src=b/'src';src.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
def run(cmd,label):
 d['commands'].append(cmd)
 with(b/(label+'.log')).open('x')as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=190)
 assert z.returncode==0,(label,(b/(label+'.log')).read_text()[-9000:])
try:
 closure=r/'results/Kimi-I192-smoke-static-closure-r8669zg.json';assert sha(closure)=='6baffc37e299a5d740644e0b911a1d3b140dd04f445c1b2ee9ea64ab6398007d';parent=json.loads(closure.read_text());assert sha(parent['native'])==parent['native_SHA256']=='6125640af7f5878e08f00f92527b3c975738bf3fc3563cb2b6afabb5dbbf6fc8'
 old=json.loads((r/'results/Kimi-I192-smoke-CPU-r8669zf.json').read_text());y=json.loads((r/'results/Kimi-I192-CPU-r8669y.json').read_text());proof=json.loads((r/'incoming/Kimi-official-WFP4-source-r8669zu.json').read_text())
 for n,h in proof['source_SHA256'].items():assert sha(r/'incoming'/n)==h;shutil.copyfile(r/'incoming'/n,src/n)
 ops=r/'candidates/rebased-5f91ad7-r3668/Ops/Kimi/InfiniOps';sdk=r/'install/hygon-kimi-5f91ad7-r3672';provider=b/'provider.o'
 run(['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-I'+str(src),'-I'+str(ops/'src'),'-I'+str(ops/'generated'),'-I'+str(sdk/'include'),'-MD','-MF',str(b/'provider.d'),'-c',str(src/'kimi_official_wfp4_W13_r8669zu.cu'),'-o',str(provider)],'provider')
 flags=old['commands'][0][:old['commands'][0].index('-MD')];flags=[flags[0],'-I'+str(src),*flags[1:]];obj=b/'moe.o';run([*flags,'-MD','-MF',str(b/'moe.d'),'-c',str(src/'kimi_k3_moe_official_wfp4_r8669zu.cpp'),'-o',str(obj)],'moe')
 link=list(old['commands'][-1]);target='/refactor/candidates/Kimi-I192-smoke-r8669zf/moe.o';assert link.count(target)==1;link[link.index(target)]=str(obj);link.insert(link.index('-o'),str(provider));native=b/'libKimi_official_WFP4_r8669zu.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(b/'link.map'))if v.startswith('-Wl,-Map,')else v for v in link];run(link,'link')
 objs={p:sha(p)for p in link if p.endswith('.o')};assert all(objs[p]==h for p,h in parent['all_linked_object_SHA256'].items()if p!=target)
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert all(v in defs for v in ['modern_kimi_official_WFP4_r8669zu','modern_kimi_official_counts_r8669zu','modern_kimi_engine_create_r2475'])
 oldU=set(subprocess.check_output(['nm','-D','--undefined-only',parent['native']],text=True).splitlines());newU=set(subprocess.check_output(['nm','-D','--undefined-only',str(native)],text=True).splitlines());assert not newU-oldU,sorted(newU-oldU)
 wrapper=b/'libKimi_full_MoE_official_WFP4_r8669zu.so';shutil.copyfile(y['library'],wrapper);needed=subprocess.check_output(['readelf','-d',str(wrapper)],text=True);assert y['native']in needed;run(['patchelf','--replace-needed',y['native'],str(native),str(wrapper)],'wrapper-rebind');needed_new=subprocess.check_output(['readelf','-d',str(wrapper)],text=True);assert str(native)in needed_new and y['native']not in needed_new
 fresh={}
 for md in b.glob('*.d'):
  paths=shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]);assert not any('/incoming/'in p for p in paths);fresh[md.name]={p:sha(p)for p in paths if pathlib.Path(p).is_file()}
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),library=str(wrapper),library_SHA256=sha(wrapper),source_SHA256=proof['source_SHA256'],all_linked_object_SHA256=objs,fresh_MD=fresh,readelf_dynamic=subprocess.check_output(['readelf','-d',str(native)],text=True),wrapper_needed=needed_new,wrapper_parent_SHA256=sha(y['library']),parent_native_SHA256=parent['native_SHA256'],new_undefined=sorted(newU-oldU),scope=proof['scope'])
except BaseException as exc:d['error']=repr(exc);raise
finally:
 p=r/'results/Kimi-official-WFP4-CPU-r8669zu.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native','native_SHA256','library','library_SHA256']}))
