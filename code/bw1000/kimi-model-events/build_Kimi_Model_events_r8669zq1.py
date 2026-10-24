import pathlib,json,hashlib,subprocess,shlex,shutil,os
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-Model-events-r8669zq1';b.mkdir(exist_ok=False);src=b/'src';src.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
def run(cmd,label):
 d['commands'].append(cmd)
 with(b/(label+'.log')).open('x')as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=190)
 assert z.returncode==0,(label,(b/(label+'.log')).read_text()[-9000:])
try:
 old=json.loads((r/'results/Kimi-I192-smoke-CPU-r8669zf.json').read_text());closure=r/'results/Kimi-I192-smoke-static-closure-r8669zg.json';assert sha(closure)=='6baffc37e299a5d740644e0b911a1d3b140dd04f445c1b2ee9ea64ab6398007d';parent=json.loads(closure.read_text());assert parent['completed']and sha(parent['native'])==parent['native_SHA256']=='6125640af7f5878e08f00f92527b3c975738bf3fc3563cb2b6afabb5dbbf6fc8'
 proof=json.loads((r/'incoming/Kimi-Model-events-source-r8669zq1.json').read_text())
 for n,h in proof['source_SHA256'].items():assert sha(r/'incoming'/n)==h;shutil.copyfile(r/'incoming'/n,src/n)
 flags=old['commands'][0][:old['commands'][0].index('-MD')];flags=[flags[0],'-I'+str(src),'-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine',*flags[1:]]
 changed={}
 for name,oldobj in [('kimi_decoder_model_events_r8669zq.cpp','/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/cached-objects/9b8c219f96772dee46fe.o'),('kimi_rank_worker_model_events_r8669zq.cpp','/refactor/candidates/kimi-compact-cache-r4974/objects/rank_worker.o')]:
  obj=b/(pathlib.Path(name).stem+'.o');md=obj.with_suffix('.d');run([*flags,'-MD','-MF',str(md),'-c',str(src/name),'-o',str(obj)],obj.stem);changed[oldobj]=str(obj)
 link=list(old['commands'][-1]);assert len(changed)==2
 for oldobj,obj in changed.items():assert link.count(oldobj)==1;link[link.index(oldobj)]=obj
 native=b/'libKimi_Model_events_r8669zq1.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(b/'link.map'))if v.startswith('-Wl,-Map,')else v for v in link];run(link,'link')
 allobjs={p:sha(p)for p in link if p.endswith('.o')};assert all(allobjs[p]==h for p,h in parent['all_linked_object_SHA256'].items()if p not in changed)
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert all(x in defs for x in ['modern_kimi_model_events_configure_r8669zq','modern_kimi_model_events_read_r8669zq','modern_kimi_model_events_disable_r8669zq','modern_kimi_engine_forward_r2475','modern_kimi_I192_counts_r8669y'])
 fresh={}
 for md in b.glob('*.d'):
  paths=shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]);assert not any('/incoming/'in p for p in paths);fresh[md.name]={p:sha(p)for p in paths if pathlib.Path(p).is_file()};assert str(src/'kimi_model_events_r8669zq.hpp')in fresh[md.name]
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),library=str(native),library_SHA256=sha(native),parent_native_SHA256=parent['native_SHA256'],parent_raw_receipt_SHA256=sha(closure),source_SHA256=proof['source_SHA256'],changed_objects=changed,all_linked_object_SHA256=allobjs,fresh_MD=fresh,readelf_dynamic=subprocess.check_output(['readelf','-d',str(native)],text=True),new_undefined=sorted(set(subprocess.check_output(['nm','-D','--undefined-only',str(native)],text=True).splitlines())-set(subprocess.check_output(['nm','-D','--undefined-only',parent['native']],text=True).splitlines())),scope=proof['scope'])
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-Model-events-CPU-r8669zq1.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native','native_SHA256']}))
