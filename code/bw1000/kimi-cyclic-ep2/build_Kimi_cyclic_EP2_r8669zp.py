import pathlib,json,hashlib,subprocess,shlex,os,shutil
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-cyclic-EP2-r8669zp';b.mkdir(exist_ok=False);src=b/'src';src.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();proof=json.loads((r/'incoming/Kimi-cyclic-EP2-source-r8669zp.json').read_text());d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
def run(cmd):
 d['commands'].append(cmd);z=subprocess.run(cmd,capture_output=True,text=True,timeout=180);assert z.returncode==0,z.stderr[-8000:]
try:
 old=json.loads((r/'results/Kimi-provider-phases-CPU-r8669zn1.json').read_text());assert old['completed']
 for n,h in proof['source_SHA256'].items():assert sha(r/'incoming'/n)==h;(src/n).write_bytes((r/'incoming'/n).read_bytes())
 hp=src/'models/kimi_k3/kimi_k3_moe.hpp';hp.parent.mkdir(parents=True);shutil.copyfile(src/'kimi_k3_moe_cyclic_EP2_r8669zp.hpp',hp)
 cmd=list(old['commands'][0]);cmd.insert(1,'-I'+str(src));cmd[cmd.index('-MF')+1]=str(b/'moe.d');cmd[cmd.index('-o')+1]=str(b/'moe.o');cmd[cmd.index('-c')+1]=str(src/'kimi_k3_moe_cyclic_EP2_r8669zp.cpp');run(cmd)
 run(['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-MMD','-MF',str(b/'mask.d'),'-c',str(src/'kimi_cyclic_EP2_mask_r8669zp.cu'),'-o',str(b/'mask.o')])
 link=list(old['commands'][2]);oldobj='/refactor/candidates/Kimi-provider-phases-r8669zn1/moe.o';oldmask='/refactor/candidates/Kimi-EP2-r8669zj/mask.o';assert link.count(oldobj)==link.count(oldmask)==1;link[link.index(oldobj)]=str(b/'moe.o');link[link.index(oldmask)]=str(b/'mask.o');native=b/'libKimi_cyclic_EP2_r8669zp.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(b/'link.map'))if x.startswith('-Wl,-Map,')else x for x in link];run(link)
 allobjs={p:sha(p)for p in link if p.endswith('.o')};assert all(allobjs[p]==h for p,h in old['all_linked_object_SHA256'].items()if p not in [oldobj,oldmask])
 cmd=list(old['commands'][3]);cmd.insert(1,'-I'+str(src));source=next(x for x in cmd if x.endswith('/kimi_TP16_provider_phases_r8669zn1.cpp'));cmd[cmd.index(source)]=str(src/'kimi_TP16_cyclic_EP2_r8669zp.cpp');cmd[cmd.index(old['native'])]=str(native);library=b/'libKimi_cyclic_EP2_full_MoE_r8669zp.so';cmd[cmd.index('-o')+1]=str(library);cmd[cmd.index('-MF')+1]=str(b/'adapter.d');run(cmd)
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);und=subprocess.check_output(['nm','-D','--undefined-only',str(native)],text=True);assert all(n in defs and n not in und for n in ['modern_kimi_CYCLIC_EP2_mask_r8669zp','modern_kimi_provider_phases_r8669zn','modern_kimi_provider_mark_r8669zn']);needed=subprocess.check_output(['readelf','-d',str(library)],text=True);assert str(native)in needed;md={}
 for p in b.glob('*.d'):
  paths=shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]);assert not any('/incoming/'in x for x in paths);md[p.name]={x:sha(x)for x in paths if pathlib.Path(x).is_file()}
  if p.name in ['moe.d','adapter.d']:assert str(hp)in md[p.name]
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),library=str(library),library_SHA256=sha(library),source_SHA256=proof['source_SHA256'],all_linked_object_SHA256=allobjs,fresh_MD=md,wrapper_needed=needed,scope='OnlyMoEhostTU+tinycyclicmask+thinwrapper; W13/W2/providerhost/SDK unchanged, realFQNstride2/evenoddmask. NoGPUinit/dlopen.')
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-cyclic-EP2-CPU-r8669zp.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native_SHA256','library_SHA256']}))
