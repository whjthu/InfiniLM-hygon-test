import pathlib,json,hashlib,subprocess,time,os,shutil,shlex
r=pathlib.Path('/refactor');b=r/'candidates/glm-pv-row-r8734';assert not b.exists();(b/'src').mkdir(parents=True)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'commands':[],'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']};tick=time.monotonic()
def run(a,n):
 q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=180);(b/(n+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError((n,p.returncode,p.stderr[-5000:]))
 return p.stdout
try:
 m=json.loads((r/'incoming/GLM-PV-row-CPU-manifest-r8734.json').read_text())
 for n,h in m['sources_SHA256'].items():assert sha(r/'incoming'/n)==h,n
 p=json.loads((r/'candidates/glm-prefix-bridge-r8705/CPU-build.json').read_text());assert p['completed']and sha(p['library'])==p['library_SHA256']=='5f3dd3d9c2a6715992be48f5de514a145c11d971cb7e07a4ec1bc10fb6a7ad52'
 prior=json.loads((r/'candidates/glm-prefill-prefix-r8703/CPU-build.json').read_text());oldcmd=next(c for c in prior['commands']if '-c'in c and any(v.endswith('/glm_prefill_prefix_dense_r8703.cpp')for v in c));flags=[];i=1
 while i<len(oldcmd):
  if oldcmd[i]in['-c','-o','-MF','-MT','-MQ']:i+=2;continue
  if oldcmd[i]in['-MD','-MMD']:i+=1;continue
  flags.append(oldcmd[i]);i+=1
 src=b/'src/glm_prefill_pv_row_view_r8734.cpp';shutil.copyfile(r/'incoming'/src.name,src);proof=b/'src/glm_prefill_prefix_proof_r8703.hpp';shutil.copyfile(r/'candidates/glm-prefill-prefix-r8703/src'/proof.name,proof);assert sha(proof)=='1f7716d44019d455201f696be2badc7008beddcf77d9efdf9afb8aad5b43ac39'
 obj=b/'pv-row.o';dep=b/'pv-row.d';run([oldcmd[0],*flags,'-MD','-MF',str(dep),'-c',str(src),'-o',str(obj)],'compile')
 link=next(list(c)for c in p['commands']if '-shared'in c);old=str(r/'candidates/glm-prefill-prefix-r8703/glm_prefill_prefix_dense_r8703.cpp.o');assert link.count(old)==1;objects={v:sha(v)for v in link if v.endswith('.o')};assert len(objects)==28
 link[link.index(old)]=str(obj);lib=b/'libglm_pv_row_r8734.so';link[link.index('-o')+1]=str(lib);run(link,'link');exports=run(['nm','-D','--defined-only',str(lib)],'exports');assert all(v in exports for v in ['glm_layer_forward_r5670','modern_glm_prefill_prefix_plan_counts_r8703','modern_glm_dense_mode_r48_set'])
 for v,h in objects.items():assert sha(v)==h
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),source=str(src),source_SHA256=sha(src),replaced_object=old,reused_objects_SHA256={v:h for v,h in objects.items()if v!=old},MD_SHA256=sha(dep),proof_header_SHA256=sha(proof),parent_library_SHA256=p['library_SHA256'],scope='One PV input row-view TU; mode2 only/defaultOFF; QK/scale/softmax/V prefix/M1/count/order unchanged; reused27 objects incl CPU169 bridge. Compile only is not GPU hit/quality/performance.')
except BaseException as e:q['error']=repr(e);raise
finally:q['seconds']=time.monotonic()-tick;(b/'CPU-build.json').write_text(json.dumps(q,indent=2))
print(json.dumps({k:q.get(k)for k in ['completed','producer','seconds','library','library_SHA256','error']}))
