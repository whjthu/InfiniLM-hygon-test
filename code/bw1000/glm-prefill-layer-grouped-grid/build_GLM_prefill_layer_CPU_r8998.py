import pathlib,json,hashlib,subprocess,os,time
r=pathlib.Path('/refactor');i=r/'incoming';b=r/'candidates/glm-prefill-layer-r8998';b.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((i/'GLM-prefill-layer-CPU-manifest-r8998.json').read_text());q={'completed':False,'GPU':False,'dlopen':False,'compiled_TUs':1,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']};start=time.perf_counter();commands=[]
def run(a,label,timeout=160):
 commands.append(a);p=subprocess.run(a,capture_output=True,text=True,timeout=timeout);(b/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.stderr[-7000:]);return p.stdout
try:
 for n,h in m['sources_SHA256'].items():assert sha(i/n)==h,n
 parent=r/'candidates/glm-hybrid-decode-r5386';proof=json.loads((parent/'build.json').read_text());assert proof['completed']and sha(parent/'libnative.so')==proof['library_sha256']=='f35956f532833ce964bf766c8a61b44b56fe79042921d86710ccd075101705d2'
 gpu={str(parent/n):sha(parent/n)for n in proof['linked_objects_sha256']};assert all(gpu[str(parent/n)]==h for n,h in proof['linked_objects_sha256'].items())
 native_parent=r/'candidates/glm-exact-budget-r8854';np=json.loads((native_parent/'CPU-build.json').read_text());assert np['completed']and sha(np['library'])==np['library_SHA256']=='68acfe089f757014e564c0d0ca5c896ac1f47cd98fa3a843df8b2dd5bf8c63c6'
 args=next(list(a)for a in np['commands']if '-shared'in a);objects={a:sha(a)for a in args if a.endswith('.o')};assert len(objects)==28
 expected={**np['reused_objects_SHA256'],**{v['object']:v['object_SHA256']for v in np['compiled']}};assert objects==expected
 helper=b/'libglm_prefill_grid_r8998.so';ha=[str(i/'glm_prefill_grid_integration_r8998.cpp')if a=='/refactor/incoming/glm_hybrid_decode_native_r5386.cpp'else a for a in proof['args']];ha.insert(1,'-Wl,-Bsymbolic-functions,-soname,libglm_prefill_grid_r8998.so');ha[ha.index('-o')+1]=str(helper);run(ha,'helper')
 lib=b/'libglm_prefill_layer_r8998.so';replace={'-L/refactor/candidates/glm-hybrid-decode-r5386':'-L'+str(b),'-lnative':'-l:libglm_prefill_grid_r8998.so','-Wl,-rpath,/refactor/candidates/glm-hybrid-decode-r5386':'-Wl,-rpath,'+str(b)};assert all(args.count(n)==1 for n in replace);args=[replace.get(a,a)for a in args];args[args.index('-o')+1]=str(lib);run(args,'native-link',30)
 hd=run(['readelf','-d',str(helper)],'helper-ELF',10);nd=run(['readelf','-d',str(lib)],'native-ELF',10);assert '(SONAME)'in hd and '[libglm_prefill_grid_r8998.so]'in hd and '[libglm_prefill_grid_r8998.so]'in nd and '[libnative.so]'not in nd
 nm=run(['nm','-D','--defined-only',str(helper)],'helper-exports',10);assert all(n in nm for n in ['glm_grouped_down_r5123','glm_prefill_grid_resolve_r8998','glm_prefill_grid_policy_r8998','glm_prefill_grid_counts_r8998','glm_certified_repair_r5248','cudaLaunchKernel'])
 nm=run(['nm','-D','--defined-only',str(lib)],'native-exports',10);assert all(n in nm for n in ['glm_layer_create_r5670','glm_layer_forward_r5670','modern_glm_model_create','modern_glm_vector_policy_r8540'])
 assert all(sha(n)==h for n,h in {**objects,**gpu}.items());q.update(completed=True,elapsed_seconds=time.perf_counter()-start,library=str(lib),library_SHA256=sha(lib),helper=str(helper),helper_SHA256=sha(helper),source_SHA256=sha(i/'glm_prefill_grid_integration_r8998.cpp'),reused_native_objects_SHA256=objects,reused_GPU_objects_SHA256=gpu,helper_ELF_dynamic=hd,native_ELF_dynamic=nd)
except BaseException as exc:q['failure']=repr(exc)
q['commands']=commands;(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:v for k,v in q.items()if k not in ['commands','reused_native_objects_SHA256','reused_GPU_objects_SHA256']}));assert q['completed']
