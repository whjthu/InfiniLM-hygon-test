import pathlib,json,hashlib,subprocess,shlex,shutil,os
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-I192-smoke-r8669zf';b.mkdir(exist_ok=False);src=b/'src';src.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
def run(cmd,label):
 d['commands'].append(cmd)
 with(b/(label+'.log')).open('x')as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=190)
 assert z.returncode==0,(label,(b/(label+'.log')).read_text()[-9000:])
try:
 parent=r/'results/Kimi-I192-guard-CPU-r8669zd.json';assert sha(parent)=='6f99924fb6b239698d3ee038a72acb99513d64679ba9aaf46832dc71538ec047';old=json.loads(parent.read_text());assert old['completed'];assert old['native_SHA256']=='13c1913ca5e6b993433ff8f07cb186d1d3ed75f374eeb7d7cf1973b293fc7b5c'
 proof=json.loads((r/'incoming/Kimi-I192-smoke-CPU-source-r8669zf.json').read_text())
 for n,h in proof['source_SHA256'].items():assert sha(r/'incoming'/n)==h;shutil.copyfile(r/'incoming'/n,src/n)
 cmd=old['commands'][0];flags=cmd[:cmd.index('-MD')];flags=[('-I'+str(src))if v=='-I/refactor/candidates/Kimi-I192-guard-r8669zd/src'else v for v in flags]
 obj=b/'moe.o';md=b/'moe.d';run([*flags,'-MD','-MF',str(md),'-c',str(src/'kimi_k3_moe_I192_caller_smoke_r8669zf.cpp'),'-o',str(obj)],'moe')
 deps=sorted(set(shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1])));d['fresh_MD']={p:sha(p)for p in deps if pathlib.Path(p).is_file()};assert all('/incoming/'not in p for p in deps)
 link=list(old['commands'][-1]);target='/refactor/candidates/Kimi-I192-guard-r8669zd/moe.o';assert link.count(target)==1;link[link.index(target)]=str(obj);native=b/'libKimi_I192_caller_smoke_r8669zf.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(b/'link.map'))if v.startswith('-Wl,-Map,')else v for v in link];run(link,'link')
 oldU=set(subprocess.check_output(['nm','-D','--undefined-only',old['native']],text=True).splitlines());newU=set(subprocess.check_output(['nm','-D','--undefined-only',str(native)],text=True).splitlines());assert not(newU-oldU),sorted(newU-oldU)
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert all(s in defs for s in ['modern_kimi_engine_create_r2475','modern_kimi_I192_counts_r8669y','modern_kimi_I192_workspace_r8669zc','modern_kimi_vocab_head_get_r8341','modern_kimi_I192_caller_smoke_r8669zf'])
 allobjs={p:sha(p)for p in link if str(p).endswith('.o')};d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),library=str(native),library_SHA256=sha(native),source_SHA256=proof['source_SHA256'],all_linked_object_SHA256=allobjs,objects={str(obj):sha(obj)},unchanged_objects={p:h for p,h in allobjs.items()if p!=str(obj)},Runtime_SHA256=old['Runtime_SHA256'],parent_native_SHA256=old['native_SHA256'],parent_raw_receipt_SHA256=sha(parent),new_undefined=sorted(newU-oldU),readelf_dynamic=subprocess.check_output(['readelf','-d',str(native)],text=True),scope=proof['scope'])
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-I192-smoke-CPU-r8669zf.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native','native_SHA256']}))
