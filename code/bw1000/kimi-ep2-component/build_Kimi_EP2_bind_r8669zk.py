import pathlib,json,hashlib,subprocess,shlex,os
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-EP2-bind-r8669zk';b.mkdir(exist_ok=False);oldb=r/'candidates/Kimi-EP2-r8669zj';p=r/'results/Kimi-EP2-CPU-r8669zj.json';old=json.loads(p.read_text());assert not old['completed']and len(old['commands'])==4
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[],'original215_FAILED_preserved':sha(p),'mechanical_fix':'Onlyadapter include search roots for unchangedprivateheader relative includes; reuse actual215 native/mask/MoE objects, no math compile.'}
try:
 cmd=list(old['commands'][-1]);native=oldb/'libKimi_EP2_r8669zj.so';assert native.is_file();model=r/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM';common=r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM';cmd[cmd.index('-I'+str(model/'csrc')):cmd.index('-I'+str(model/'csrc'))]=['-I'+str(model/'csrc/models/kimi_k3'),'-I'+str(common/'csrc/models/kimi_k3')];adapter=b/'libKimi_EP2_full_MoE_r8669zk.so';cmd[cmd.index('-o')+1]=str(adapter);cmd[cmd.index('-MF')+1]=str(b/'adapter.d');d['commands'].append(cmd)
 z=subprocess.run(cmd,capture_output=True,text=True,timeout=180);assert z.returncode==0,z.stderr[-9000:]
 allobjs={p:sha(p)for p in old['commands'][2]if p.endswith('.o')};assert allobjs['/refactor/candidates/kimi-packed-reuse-r6136/packed.o']=='355607f22e0232b6422cef6f81224af24f62b585e5e251fbb656e7eb0da4700c'
 d['fresh_MD']={}
 for md in [oldb/'moe.d',oldb/'mask.d',b/'adapter.d']:
  paths=shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]);d['fresh_MD'][str(md)]={p:sha(p)for p in paths if pathlib.Path(p).is_file()};assert all('/incoming/'not in p for p in paths)
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);und=subprocess.check_output(['nm','-D','--undefined-only',str(native)],text=True);assert all(x in defs and x not in und for x in ['modern_kimi_EP2_mask_r8669zj','modern_kimi_EP2_counts_r8669zj','modern_kimi_grouped_r4620','modern_kimi_grouped_I192_r8669y'])
 needed=subprocess.check_output(['readelf','-d',str(adapter)],text=True);assert str(native)in needed
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),library=str(adapter),library_SHA256=sha(adapter),source_SHA256=json.loads((r/'incoming/Kimi-EP2-source-r8669zj.json').read_text())['source_SHA256'],all_linked_object_SHA256=allobjs,native_defined=defs,native_undefined=und,wrapper_needed=needed)
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-EP2-bind-CPU-r8669zk.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native_SHA256','library_SHA256']}))
