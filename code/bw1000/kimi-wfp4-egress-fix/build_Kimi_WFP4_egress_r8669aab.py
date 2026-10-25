import pathlib,json,hashlib,subprocess,shutil,shlex,os
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-WFP4-egress-r8669aab';b.mkdir(exist_ok=False);src=b/'src';src.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
def run(cmd,label):
 d['commands'].append(cmd)
 with(b/(label+'.log')).open('x')as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=190)
 assert z.returncode==0,(label,(b/(label+'.log')).read_text()[-6000:])
try:
 oldp=r/'results/Kimi-official-WFP4-CPU-r8669zu.json';old=json.loads(oldp.read_text());parent=json.loads((r/'results/Kimi-official-WFP4-wrapper-CPU-r8669zv.json').read_text());assert parent['completed']and sha(parent['native'])==parent['native_SHA256']=='6b669a1a5f3e58caaaf78d441931e7098f56bb94cb80e558911fa5b402544d1d'
 proof=json.loads((r/'incoming/Kimi-WFP4-egress-source-r8669aab.json').read_text());name='kimi_official_WFP4_egress_r8669aab.cu';assert sha(r/'incoming'/name)==proof['source_SHA256'][name];shutil.copyfile(r/'incoming'/name,src/name)
 original_source=pathlib.Path(old['commands'][0][old['commands'][0].index('-c')+1]);assert sha(original_source)==proof['parent_provider_source_SHA256'];a=original_source.read_text();z=(src/name).read_text();assert a.replace('ro*8+colid*2+ri','ro*8+ri*4+colid')==z
 cmd=list(old['commands'][0]);cmd[cmd.index(str(original_source))]=str(src/name);oldobj=cmd[cmd.index('-o')+1];newobj=b/'provider.o';cmd[cmd.index('-o')+1]=str(newobj);cmd[cmd.index('-MF')+1]=str(b/'provider.d');run(cmd,'provider')
 link=list(old['commands'][2]);assert link.count(oldobj)==1;link[link.index(oldobj)]=str(newobj);native=b/'libKimi_WFP4_egress_r8669aab.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(b/'link.map'))if v.startswith('-Wl,-Map,')else v for v in link];run(link,'link')
 objs={v:sha(v)for v in link if v.endswith('.o')};assert all(objs[v]==h for v,h in parent['all_linked_object_SHA256'].items()if v!=oldobj)
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert 'modern_kimi_official_WFP4_r8669zu'in defs
 und=lambda p:set(subprocess.check_output(['nm','-D','--undefined-only',str(p)],text=True).splitlines());assert not und(native)-und(parent['native'])
 # Recompile the unchanged thin host wrapper only to bind actual new DT_NEEDED.
 oldwrap=json.loads((r/'results/Kimi-official-WFP4-pair-CPU-r8669zx.json').read_text());assert oldwrap['completed'];cmd=list(oldwrap['commands'][0]);assert cmd.count(parent['native'])==1;cmd[cmd.index(parent['native'])]=str(native);wrapper=b/'libKimi_TP16_WFP4_egress_pair_r8669aab.so';cmd[cmd.index('-o')+1]=str(wrapper);cmd[cmd.index('-MF')+1]=str(b/'adapter.d');run(cmd,'adapter-rebind');needed=subprocess.check_output(['readelf','-d',str(wrapper)],text=True);assert str(native)in needed and parent['native']not in needed
 fresh={}
 for md in b.glob('*.d'):
  paths=shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]);fresh[md.name]={p:sha(p)for p in paths if pathlib.Path(p).is_file()}
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),library=str(wrapper),library_SHA256=sha(wrapper),source_SHA256=proof['source_SHA256'],all_linked_object_SHA256=objs,fresh_MD=fresh,wrapper_needed=needed,parent_native_SHA256=parent['native_SHA256'],new_undefined=[],new_GPU_TUs=1,unchanged_thin_host_wrapper_rebound=True,change='Only MMA rawW13 accumulator egress column ro*8+colid*2+ri -> ro*8+ri*4+colid; no other math/provider/Runtime/SDK edits; GPU not run.')
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-WFP4-egress-CPU-r8669aab.json';assert not p.exists();p.write_bytes((json.dumps(d,indent=2)+'\n').encode());print(json.dumps({k:d.get(k)for k in ['completed','step','error','native','native_SHA256','library','library_SHA256']}))
