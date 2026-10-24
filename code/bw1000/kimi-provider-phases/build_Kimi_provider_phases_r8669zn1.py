import pathlib,json,hashlib,subprocess,shlex,os
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-provider-phases-r8669zn1';b.mkdir(exist_ok=False);src=b/'src';src.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();proof=json.loads((r/'incoming/Kimi-provider-phases-source-r8669zn1.json').read_text());d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
def run(cmd):
 d['commands'].append(cmd);z=subprocess.run(cmd,capture_output=True,text=True,timeout=180);assert z.returncode==0,z.stderr[-8000:]
try:
 old=json.loads((r/'results/Kimi-owner-events-CPU-r8669zm.json').read_text());assert old['completed']
 for n,h in proof['source_SHA256'].items():assert sha(r/'incoming'/n)==h;(src/n).write_bytes((r/'incoming'/n).read_bytes())
 cmd=list(old['commands'][0]);cmd[cmd.index('-MF')+1]=str(b/'moe.d');cmd[cmd.index('-o')+1]=str(b/'moe.o');cmd[cmd.index('-c')+1]=str(src/'kimi_k3_moe_provider_phases_r8669zn1.cpp');run(cmd)
 run(['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-MMD','-MF',str(b/'provider.d'),'-c',str(src/'kimi_provider_phase_launch_r8669zn1.cu'),'-o',str(b/'provider-host.o')])
 link=list(old['commands'][1]);oldobj='/refactor/candidates/Kimi-owner-events-r8669zm/moe.o';assert link.count(oldobj)==1;link[link.index(oldobj)]=str(b/'moe.o');link.insert(link.index('-o'),str(b/'provider-host.o'));native=b/'libKimi_provider_phases_r8669zn1.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(b/'link.map'))if x.startswith('-Wl,-Map,')else x for x in link];run(link)
 allobjs={p:sha(p)for p in link if p.endswith('.o')};assert all(allobjs[p]==h for p,h in old['all_linked_object_SHA256'].items()if p!=oldobj)
 cmd=list(old['commands'][2]);source=next(x for x in cmd if x.endswith('/kimi_TP16_owner_events_r8669zm.cpp'));cmd[cmd.index(source)]=str(src/'kimi_TP16_provider_phases_r8669zn1.cpp');cmd[cmd.index(old['native'])]=str(native);library=b/'libKimi_provider_phases_full_MoE_r8669zn1.so';cmd[cmd.index('-o')+1]=str(library);cmd[cmd.index('-MF')+1]=str(b/'adapter.d');run(cmd)
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);und=subprocess.check_output(['nm','-D','--undefined-only',str(native)],text=True);assert 'modern_kimi_provider_phases_r8669zn'in defs and 'modern_kimi_provider_mark_r8669zn'in defs;oldU=set(subprocess.check_output(['nm','-D','--undefined-only',old['native']],text=True).splitlines());newU=set(und.splitlines())-oldU;assert not any(any(v in line for v in ['packRoutes','buildJobs','groupedW13','staticW2','provider_mark'])for line in newU),('new_hostlaunch_U',sorted(newU));needed=subprocess.check_output(['readelf','-d',str(library)],text=True);assert str(native)in needed
 md={}
 for p in b.glob('*.d'):
  paths=shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]);assert not any('/incoming/'in x for x in paths);md[p.name]={x:sha(x)for x in paths if pathlib.Path(x).is_file()}
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),library=str(library),library_SHA256=sha(library),source_SHA256=proof['source_SHA256'],all_linked_object_SHA256=allobjs,fresh_MD=md,wrapper_needed=needed,new_undefined=sorted(newU),provider_host_symbols=subprocess.check_output(['nm','-C',str(b/'provider-host.o')],text=True),scope='1MoEhostTU+thinwrapper+hostonlyexternalGPUlaunchstubTU; unchangedallGPU/provider/maskobjects/SDK. NoGPUinit/dlopen.')
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-provider-phases-CPU-r8669zn1.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native_SHA256','library_SHA256']}))
