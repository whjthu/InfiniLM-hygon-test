import hashlib,json,pathlib,shutil,subprocess,time,os,shlex
r=pathlib.Path('/refactor');b=r/'candidates/glm-prefill-prefix-r8703';assert not b.exists();(b/'src').mkdir(parents=True);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((r/'incoming/GLM-prefill-prefix-CPU-manifest-r8703.json').read_text())
for n,h in m['sources_SHA256'].items():assert sha(r/'incoming'/n)==h,n
parent=r/'candidates/glm-shared-validation-model-r5817/CPU-build.json';p=json.loads(parent.read_text());assert p['completed']and sha(p['library'])==p['library_SHA256']=='1af0b95df9db1c68f9e8c00124dbb0b331cd1b3650afd4506e3befa30f98be9d'
a=json.loads((r/'candidates/glm-transient-layer-r5681/CPU-build.json').read_text());commands=[];report={'completed':False,'CPU_only':True,'dlopen':False,'GPU_initialized':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'parent_DSO_SHA256':sha(p['library'])};start=time.monotonic()
def run(cmd,name):
 commands.append(cmd);t=subprocess.run(cmd,text=True,capture_output=True);(b/(name+'.log')).write_text(t.stdout+t.stderr)
 if t.returncode:raise RuntimeError((name,t.returncode,t.stderr[-6000:]))
 return t.stdout
try:
 shutil.copyfile(r/'incoming/glm_prefill_prefix_proof_r8703.hpp',b/'src/glm_prefill_prefix_proof_r8703.hpp')
 link=next(list(v)for v in p['commands']if '-shared'in v);objects={v:sha(v)for v in link if v.endswith('.o')};assert len(objects)==28
 replacements={};compiled=[]
 for name,local,oldobj,source_sha in [('glm_moe_dsa_projected_cache_view_r927.cpp','glm_prefill_prefix_cache_r8703.cpp',str(r/'candidates/glm-shared-validation-model-r5817/bridge.o'),'4d7f4a9293cff492c1e8835560d345ea7aeace58d11b2b303188b312a0934962'),('glm_moe_dsa_dense_modes_r48.cpp','glm_prefill_prefix_dense_r8703.cpp',str(r/'candidates/glm-transient-layer-r5681/private-13.o'),'5d27978cced7d942a0f91ee8233aae30a29198cc8f494779db788e3f88f4a435')]:
  rec=next(v for v in a['compiled']if pathlib.Path(v['source']).name==name);assert oldobj in objects
  actual=r/'candidates/glm-shared-validation-model-r5817/src/glm_shared_validation_cache_r5814.cpp' if name.startswith('glm_moe_dsa_projected_cache') else pathlib.Path(rec['source']);assert sha(actual)==source_sha
  oldcmd=next(v for v in a['commands']if '-c'in v and rec['source']in v);src=b/'src'/local;shutil.copyfile(r/'incoming'/local,src);obj=b/(local+'.o');dep=b/(local+'.d');fl=[];i=1
  while i<len(oldcmd):
   if oldcmd[i]in['-c','-o','-MF','-MT','-MQ']:i+=2;continue
   if oldcmd[i]in['-MD','-MMD']:i+=1;continue
   fl.append(oldcmd[i]);i+=1
  run([oldcmd[0],*fl,'-MD','-MF',str(dep),'-c',str(src),'-o',str(obj)],local)
  deps=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]);proof=str(b/'src/glm_prefill_prefix_proof_r8703.hpp');assert proof in deps and sha(proof)==m['sources_SHA256']['glm_prefill_prefix_proof_r8703.hpp']
  replacements[oldobj]=str(obj);compiled.append({'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'replaces':oldobj,'base_source':str(actual),'base_source_SHA256':source_sha,'MD_SHA256':sha(dep),'new_proof_header_SHA256':sha(proof),'compile_dependency_count':len(deps)})
 link=[replacements.get(v,v)for v in link];lib=b/'libglm_prefill_prefix_r8703.so';link[link.index('-o')+1]=str(lib);run(link,'link');exports=run(['nm','-D','--defined-only',str(lib)],'exports');assert all(v in exports for v in['modern_glm_prefill_prefix_plan_counts_r8703','modern_glm_dense_mode_r48_set','modern_glm_shared_validation_counts_r5814'])
 for n,h in objects.items():assert sha(n)==h
 report.update(completed=True,compiled=compiled,library=str(lib),library_SHA256=sha(lib),reused_objects_SHA256={n:h for n,h in objects.items()if n not in replacements},scope='2 existing CPU TUs + private sealed proof header, reused26 objects. DefaultOFF actual CPUplan provenance only; same numeric body/sync/stream/SDK. No dlopen/GPU/weights. Native compile not GPU scope/quality/performance proof.')
except BaseException as e:report['error']=repr(e);raise
finally:report.update(seconds=time.monotonic()-start,commands=commands);(b/'CPU-build.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k]for k in ['completed','producer','seconds','library_SHA256','scope']}))
