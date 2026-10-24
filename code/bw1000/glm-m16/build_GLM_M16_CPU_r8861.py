import pathlib,json,hashlib,subprocess,shutil,shlex,os,time
r=pathlib.Path('/refactor');b=r/'candidates/glm-M16-r8861';assert not b.exists();b.mkdir();begin=time.monotonic()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[],'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
def run(a,n):
 q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=200);(b/(n+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(n,p.stderr[-5000:]);return p.stdout
def deps(p):return shlex.split(pathlib.Path(p).read_text().replace('\\\n',' ').split(':',1)[1])
def mirror(p):return b/'tree'/pathlib.Path(p).relative_to(r)
try:
 mp=r/'incoming/GLM-M16-CPU-manifest-r8861.json';m=json.loads(mp.read_text())
 for n,h in m['sources_SHA256'].items():assert sha(r/'incoming'/n)==h,n
 prior=json.loads((r/'candidates/glm-exact-budget-r8854/CPU-build.json').read_text());assert prior['completed']and sha(prior['library'])==prior['library_SHA256']==m['parent_DSO_SHA256']
 link=next(list(x)for x in prior['commands']if '-shared'in x);objects={v:sha(v)for v in link if v.endswith('.o')};assert len(objects)==28
 records=[]
 for rec in m['bound_object_map']:
  old=rec['object'];assert old in objects
  if rec.get('resident_header_dependencies')or old=='/refactor/candidates/glm-vector-model-r8542/vector-1.o':
   assert sha(old)==rec['object_SHA256'] and sha(pathlib.Path(old).with_suffix('.d'))==rec['MD_SHA256'];records.append(rec)
 assert len(records)==7
 allcmd=[]
 for folder in sorted({str(pathlib.Path(v['object']).parent)for v in records}):
  doc=json.loads((pathlib.Path(folder)/'CPU-build.json').read_text());assert doc['completed'];allcmd.extend(x for x in doc['commands']if isinstance(x,list))
 header_sha={};source_sha={}
 for rec in records:
  old=rec['object'];dd=deps(pathlib.Path(old).with_suffix('.d'))
  for v in dd:
   p=pathlib.Path(v)
   if (v.startswith('/refactor/candidates/')or v.startswith('/refactor/incoming/'))and p.suffix in ('.hpp','.h'):
    dst=mirror(p);dst.parent.mkdir(parents=True,exist_ok=True);header_sha[v]=sha(p)
    if p.name=='glm_model_resident_policy_r5640.hpp':assert sha(p)=='63ef7f9ceee8f51ea52b582f71d1274c9268217e4560d87d04c9e891c070246c';shutil.copyfile(r/'incoming/GLM_M16_policy_r8861.hpp',dst)
    else:shutil.copyfile(p,dst)
  src=pathlib.Path(rec['source']);assert sha(src)==rec['source_SHA256'];dst=mirror(src);dst.parent.mkdir(parents=True,exist_ok=True);source_sha[str(src)]=sha(src)
  ov={'glm_prefill_prefix_cache_r8703.cpp':'GLM_M16_prefix_r8861.cpp','glm_vector_dispatch_r8540.cpp':'GLM_M16_vector_dispatch_r8861.cpp'}.get(src.name)
  shutil.copyfile(r/'incoming'/ov if ov else src,dst)
 replaced={};compiled=[]
 for j,rec in enumerate(records):
  old=rec['object'];cmd=next(x for x in allcmd if '-c'in x and '-o'in x and x[x.index('-o')+1]==old);src=mirror(rec['source']);flags=[];i=1
  while i<len(cmd):
   if cmd[i]in ('-c','-o','-MF','-MT','-MQ'):i+=2;continue
   if cmd[i]in ('-MD','-MMD'):i+=1;continue
   v=cmd[i]
   if v.startswith('-I/refactor/candidates/')or v=='-I/refactor/incoming':v='-I'+str(mirror(v[2:]))
   flags.append(v);i+=1
  obj=b/f'private-{j}.o';md=b/f'private-{j}.d';a=[cmd[0],'-I'+str(src.parent),*flags,'-MD','-MF',str(md),'-c',str(src),'-o',str(obj)];run(a,f'compile-{j}')
  actual=deps(md);pol=[p for p in actual if pathlib.Path(p).name=='glm_model_resident_policy_r5640.hpp']
  if rec.get('resident_header_dependencies'):assert pol and all(v.startswith(str(b)+'/tree/')and sha(v)==sha(r/'incoming/GLM_M16_policy_r8861.hpp')for v in pol)
  replaced[old]=str(obj);compiled.append({'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'replaces':old,'MD_SHA256':sha(md),'policy_dependencies_SHA256':{v:sha(v)for v in pol}})
 # Execute actual header/dispatch expressions on CPU; kernel externs are local test stubs.
 cd=b/'CPU-contract';cd.mkdir();shutil.copyfile(r/'incoming/check_GLM_M16_r8861.cpp',cd/'check.cpp');shutil.copyfile(r/'incoming/GLM_M16_policy_r8861.hpp',cd/'glm_model_resident_policy_r5640.hpp');shutil.copyfile(r/'incoming/GLM_M16_vector_dispatch_r8861.cpp',cd/'glm_vector_dispatch_r8540.cpp')
 vector_rec=next(v for v in records if v['source'].endswith('glm_vector_dispatch_r8540.cpp'));vh=next(v for v in deps(pathlib.Path(vector_rec['object']).with_suffix('.d'))if pathlib.Path(v).name=='glm_vector_dispatch_r8540.hpp');shutil.copyfile(vh,cd/'glm_vector_dispatch_r8540.hpp')
 compiler=next(x for x in allcmd if '-c'in x)[0];run([compiler,'-std=c++17','-O2','-pthread',str(cd/'check.cpp'),'-o',str(cd/'check')],'contract-build');contract=json.loads(run([str(cd/'check')],'contract-run'))
 link=[replaced.get(v,v)for v in link];lib=b/'libglm_M16_r8861.so';link[link.index('-o')+1]=str(lib);run(link,'link')
 syms=run(['nm','-D','--defined-only',str(lib)],'exports');oldsyms=run(['nm','-D','--defined-only',prior['library']],'oldexports');required={v.split()[-1]for v in oldsyms.splitlines()if v.split()and v.split()[-1].startswith(('modern_glm_','glm_'))};assert required<={v.split()[-1]for v in syms.splitlines()if v.split()}
 assert all(sha(o)==h for o,h in objects.items())
 q.update(completed=True,compiled_TUs=7,compiled=compiled,reused_objects_SHA256={p:h for p,h in objects.items()if p not in replaced},library=str(lib),library_SHA256=sha(lib),preserved_CAPI_symbols=sorted(required),CPU_contract=contract,source_manifest_SHA256=sha(mp),original_private_headers_SHA256=header_sha,original_sources_SHA256=source_sha,scope=m['scope'],bounded_attention={'existing_helper_object':'/refactor/candidates/glm-transient-layer-r5681/private-18.o','source_SHA256':'2390bf53024ea92e119aa8fdebcdeb2731d12d3f3050368aa5576c0f7c5b2a79','B':[8,16],'P_max':640,'backend':1,'M16_explicit_held_tensors_logical_bytes_max':1114112,'pool_and_operator_workspace_peak':'UNKNOWN'})
except BaseException as e:q['error']=repr(e)
finally:q['seconds']=time.monotonic()-begin;(b/'CPU-build.json').write_text(json.dumps(q,indent=2))
print(json.dumps({k:q.get(k)for k in ['completed','producer','seconds','compiled_TUs','library_SHA256','CPU_contract','error']}));assert q['completed']
