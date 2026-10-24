import pathlib,json,hashlib,subprocess,shlex,shutil,os
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-TP16-head-r8669z';b.mkdir(exist_ok=False);src=b/'src';src.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
def run(cmd,label):
 d['commands'].append(cmd)
 with(b/(label+'.log')).open('x')as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=190)
 assert z.returncode==0,(label,(b/(label+'.log')).read_text()[-9000:])
try:
 basep=r/'results/Kimi-I192-CPU-r8669y.json';assert sha(basep)=='e596e714cc9aff69a01810a27028193af402bb75bc70d4a17cda41b8d1cd2916';old=json.loads(basep.read_text());assert old['completed'];proof=json.loads((r/'incoming/Kimi-TP16-head-source-r8669z.json').read_text())
 for n,h in proof['source_SHA256'].items():
  actualname=proof['incoming_alias'][n];assert sha(r/'incoming'/actualname)==h,n;shutil.copyfile(r/'incoming'/actualname,src/n)
 original=json.loads((r/'results/Kimi-head-Model-CPU-r8344.json').read_text());cmd=original['commands'][0];assert '-c'in cmd;flags=cmd[:cmd.index('-MD')];flags=[('-I'+str(src))if x=='-I/refactor/incoming'else x for x in flags]
 obj=b/'head.o';md=b/'head.d';run([*flags,'-MD','-MF',str(md),'-c',str(src/'kimi_head_model_r8344.cpp'),'-o',str(obj)],'head')
 deps=sorted(set(shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1])));d['fresh_MD']={p:sha(p)for p in deps if pathlib.Path(p).is_file()};assert all('/incoming/'not in p for p in deps)
 actual='/refactor/candidates/kimi-vocab-Model-CPU-r8344/kimi_head_model_r8344.o';assert old['unchanged_objects'][actual]=='de7836e6220646cf46bfb120142152a105477ec1bab40deb78fedb17bdfa0534'
 link=next(list(c)for c in old['commands']if '-o'in c and c[c.index('-o')+1]==old['native']);assert link.count(actual)==1;link[link.index(actual)]=str(obj);native=b/'libKimi_TP16_I192_head_r8669z.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(b/'link.map'))if x.startswith('-Wl,-Map,')else x for x in link];run(link,'link')
 oldU=set(subprocess.check_output(['nm','-D','--undefined-only',old['native']],text=True).splitlines());newU=set(subprocess.check_output(['nm','-D','--undefined-only',str(native)],text=True).splitlines());assert not(newU-oldU),sorted(newU-oldU)
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert all(v in defs for v in ['modern_kimi_engine_create_r2475','modern_kimi_engine_forward_r2475','modern_kimi_I192_counts_r8669y','modern_kimi_vocab_head_get_r8341','DualVocabHeadR8341'])
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),library=str(native),library_SHA256=sha(native),source_SHA256=proof['source_SHA256'],objects={**old['objects'],str(obj):sha(obj)},unchanged_objects={p:h for p,h in old['unchanged_objects'].items()if p!=actual},Runtime_SHA256=old['Runtime_SHA256'],parent_native_SHA256=old['native_SHA256'],parent_raw_receipt_SHA256=sha(basep),new_undefined=sorted(newU-oldU),native_defined=defs,readelf_dynamic=subprocess.check_output(['readelf','-d',str(native)],text=True),head_scope=proof['scope'])
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-TP16-head-CPU-r8669z.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native','native_SHA256']}))
