"""One bridge TU only, on CPU. Reuse all27 actual successful8703 objects."""
import pathlib,json,hashlib,subprocess,time,os,shlex,shutil
r=pathlib.Path('/refactor');b=r/'candidates/glm-prefix-bridge-r8705';assert not b.exists();(b/'src').mkdir(parents=True)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'commands':[],'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']};tick=time.monotonic()
def run(a,name):
 q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=180);(b/(name+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError((name,p.returncode,p.stderr[-5000:]))
 return p.stdout
try:
 m=json.loads((r/'incoming/GLM-prefix-bridge-source-r8705.json').read_text());meta=r/'incoming/GLM513-actual-bridge-r8705.json';assert sha(meta)==m['source_metadata_SHA256'];frozen=json.loads(meta.read_text())
 parent=r/'candidates/glm-prefill-prefix-r8703/CPU-build.json';p=json.loads(parent.read_text());assert p['completed']and sha(p['library'])==p['library_SHA256']=='4572a7ad82a7977da19dc53d5962690c82e378834fa08411f9cd2c075fee1bd3'
 actual=r/'candidates/glm-window-model-r5797/src/glm_layer_bridge_r5670.cpp';assert sha(actual)==m['old_actual_bridge_SHA256']=='3ca3f9a68e96bfae6529273c93863bb220025eb1d2cb160da156c50740a784cb'
 src=r/'incoming/glm_prefix_layer_bridge_r8705.cpp';assert sha(src)==m['source_SHA256'][src.name];local=b/'src'/src.name;shutil.copyfile(src,local)
 oldcmd=frozen['window']['bridge_compile'];assert len(oldcmd)==1;oldcmd=oldcmd[0];flags=[];i=1
 while i<len(oldcmd):
  if oldcmd[i]in['-c','-o','-MF','-MT','-MQ']:i+=2;continue
  if oldcmd[i]in['-MD','-MMD']:i+=1;continue
  flags.append(oldcmd[i]);i+=1
 obj=b/'bridge.o';md=b/'bridge.d';run([oldcmd[0],*flags,'-MD','-MF',str(md),'-c',str(local),'-o',str(obj)],'compile')
 link=next(list(c)for c in p['commands']if '-shared'in c);oldobj=str(r/'candidates/glm-window-model-r5797/window-1.o');assert link.count(oldobj)==1
 original_objects={v:sha(v)for v in link if v.endswith('.o')};assert len(original_objects)==28 and original_objects[oldobj]=='4f1b2c067b5d2ff6a356478be28c2b3066d701057653a6c325a27664e4b40e40'
 link[link.index(oldobj)]=str(obj);lib=b/'libglm_prefix_bridge_r8705.so';link[link.index('-o')+1]=str(lib);run(link,'link')
 deps=shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]);defined=run(['nm','-D','--defined-only',str(lib)],'exports');assert all(n in defined for n in ['glm_layer_create_r5670','glm_layer_forward_r5670','modern_glm_prefill_prefix_plan_counts_r8703'])
 for f,h in original_objects.items():assert sha(f)==h
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),source=str(local),source_SHA256=sha(local),MD_SHA256=sha(md),dependencies={f:sha(f)for f in deps},replaced_object=oldobj,reused_objects_SHA256={f:h for f,h in original_objects.items()if f!=oldobj},parent_library_SHA256=p['library_SHA256'],scope=m['scope'],actual_TP_world=8,logical_layer_stage=1,logical_PP_identity=2)
except BaseException as e:q['error']=repr(e);raise
finally:q['seconds']=time.monotonic()-tick;(b/'CPU-build.json').write_text(json.dumps(q,indent=2))
print(json.dumps({k:q.get(k)for k in ['completed','producer','seconds','library','library_SHA256','error']}))
