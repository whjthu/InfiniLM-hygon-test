import pathlib,json,hashlib,subprocess,shlex,os
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-owner-events-r8669zm';b.mkdir(exist_ok=False);src=b/'src';src.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();proof=json.loads((r/'incoming/Kimi-owner-events-source-r8669zm.json').read_text());d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
def run(cmd):
 d['commands'].append(cmd);z=subprocess.run(cmd,capture_output=True,text=True,timeout=180);assert z.returncode==0,z.stderr[-8000:]
try:
 old=json.loads((r/'results/Kimi-EP2-CPU-r8669zj.json').read_text());binding=json.loads((r/'results/Kimi-EP2-bind-CPU-r8669zk.json').read_text());assert binding['completed']
 for n,h in proof['source_SHA256'].items():assert sha(r/'incoming'/n)==h;(src/n).write_bytes((r/'incoming'/n).read_bytes())
 cmd=list(old['commands'][0]);cmd[cmd.index('-MF')+1]=str(b/'moe.d');cmd[cmd.index('-o')+1]=str(b/'moe.o');cmd[cmd.index('-c')+1]=str(src/'kimi_k3_moe_owner_events_r8669zm.cpp');run(cmd)
 link=list(old['commands'][2]);oldobj='/refactor/candidates/Kimi-EP2-r8669zj/moe.o';assert link.count(oldobj)==1;link[link.index(oldobj)]=str(b/'moe.o');native=b/'libKimi_owner_events_r8669zm.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(b/'link.map'))if x.startswith('-Wl,-Map,')else x for x in link];run(link)
 allobjs={p:sha(p)for p in link if p.endswith('.o')};assert all(allobjs[p]==h for p,h in binding['all_linked_object_SHA256'].items()if p!=oldobj)
 cmd=list(binding['commands'][0]);source=next(x for x in cmd if x.endswith('/kimi_TP16_EP2_full_MoE_r8669zj.cpp'));cmd[cmd.index(source)]=str(src/'kimi_TP16_owner_events_r8669zm.cpp');cmd[cmd.index(binding['native'])]=str(native);library=b/'libKimi_owner_events_full_MoE_r8669zm.so';cmd[cmd.index('-o')+1]=str(library);cmd[cmd.index('-MF')+1]=str(b/'adapter.d');run(cmd)
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert all(x in defs for x in ['modern_kimi_owner_events_prepare_r8669zm','modern_kimi_owner_events_arm_r8669zm','modern_kimi_owner_events_read_r8669zm','modern_kimi_owner_events_close_r8669zm']);needed=subprocess.check_output(['readelf','-d',str(library)],text=True);assert str(native)in needed
 md={}
 for p in b.glob('*.d'):
  paths=shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]);assert not any('/incoming/'in x for x in paths);md[p.name]={x:sha(x)for x in paths if pathlib.Path(x).is_file()}
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),library=str(library),library_SHA256=sha(library),source_SHA256=proof['source_SHA256'],all_linked_object_SHA256=allobjs,fresh_MD=md,wrapper_needed=needed,scope='Only1MoEhostTU+thinwrapper; unchangedallGPU/provider/maskobjects/SDK. NoGPUinit/dlopen.')
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-owner-events-CPU-r8669zm.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native_SHA256','library_SHA256']}))
