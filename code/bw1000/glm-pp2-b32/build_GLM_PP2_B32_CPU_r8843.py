import pathlib,json,hashlib,subprocess,shutil,shlex,os,time
r=pathlib.Path('/refactor');b=r/'candidates/glm-PP2-B32-r8843';assert not b.exists();b.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();begin=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[],'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
def run(a,n):
 q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=200);(b/(n+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(n,p.stderr[-5000:]);return p.stdout
def mirror(p):return b/'tree'/pathlib.Path(p).relative_to(r)
def deps(s):return set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]))
try:
 mp=r/'incoming/GLM-PP2-B32-CPU-manifest-r8843.json';m=json.loads(mp.read_text())
 for n,h in m['sources_SHA256'].items():assert sha(r/'incoming'/n)==h,n
 inputs=json.loads((r/'incoming/GLM-PP2-B32-build-input-r8843.json').read_text());cpu=json.loads((r/'candidates/glm-prefix-bridge-r8705/CPU-build.json').read_text());assert cpu['completed']and sha(cpu['library'])==cpu['library_SHA256']=='5f3dd3d9c2a6715992be48f5de514a145c11d971cb7e07a4ec1bc10fb6a7ad52'
 link=next(list(c)for c in cpu['commands']if '-shared'in c);objects={v:sha(v)for v in link if v.endswith('.o')};assert len(objects)==28
 # Derive affected consumers from each currently linked object's actual MD.
 records=[]
 for rec in inputs['objects']:
  ob=rec['object'];assert ob in objects and objects[ob]==rec['SHA256'];dep=pathlib.Path(ob).with_suffix('.d')
  if dep.exists():
   dd=deps(dep.read_text());hits=[v for v in dd if pathlib.Path(v).name=='glm_model_resident_policy_r5640.hpp']
   if hits:assert len(rec.get('source',[]))==1;records.append({**rec,'actual_MD':sorted(dd)})
 assert len(records)==6,[(v['object'],v['source'])for v in records]
 overrides=inputs['overrides'];classheaders={'glm_moe_dsa_projected_cache_r22.hpp','glm_moe_dsa_projected_decoder_r22.hpp','glm_moe_dsa_projected_text_model_r22.hpp'};headers={}
 for rec in records:
  for v in rec['actual_MD']:
   p=pathlib.Path(v)
   if v.startswith('/refactor/candidates/')and '/src/'in v and p.suffix in ['.hpp','.h']:
    dst=mirror(p);dst.parent.mkdir(parents=True,exist_ok=True);headers[v]=sha(p)
    if p.name in overrides:
     ov=overrides[p.name];src=r/'incoming'/ov['incoming'];assert sha(src)==ov['SHA256']
     if p.name in classheaders:assert sha(p)==ov['SHA256'],('classheader changed',v)
     shutil.copyfile(src,dst)
    else:shutil.copyfile(p,dst)
  src=pathlib.Path(rec['source'][0]);dst=mirror(src);dst.parent.mkdir(parents=True,exist_ok=True)
  if src.name in overrides:shutil.copyfile(r/'incoming'/overrides[src.name]['incoming'],dst)
  else:assert sha(src)==inputs['unchanged_source_SHA256'][src.name];shutil.copyfile(src,dst)
  # New helper and all shared private definitions win quoted lookup in every TU.
  for n,ov in overrides.items():
   if n.endswith(('.hpp','.h')):shutil.copyfile(r/'incoming'/ov['incoming'],dst.parent/n)
 allcmd=[]
 for folder in sorted({str(pathlib.Path(rec['object']).parent)for rec in records}):
  doc=json.loads((pathlib.Path(folder)/'CPU-build.json').read_text());allcmd.extend(c for c in doc['commands']if isinstance(c,list))
 replaced={};compiled=[]
 for j,rec in enumerate(records):
  old=rec['object'];cmd=next(c for c in allcmd if '-c'in c and '-o'in c and c[c.index('-o')+1]==old);src=mirror(pathlib.Path(rec['source'][0]));flags=[];k=1
  while k<len(cmd):
   if cmd[k]in ['-c','-o','-MF','-MT','-MQ']:k+=2;continue
   if cmd[k]in ['-MD','-MMD']:k+=1;continue
   v=cmd[k]
   if v.startswith('-I/refactor/candidates/')and '/src'in v:v='-I'+str(mirror(pathlib.Path(v[2:])))
   flags.append(v);k+=1
  flags=['-I'+str(src.parent),*flags];obj=b/f'private-{j}.o';dep=b/f'private-{j}.d';run([cmd[0],*flags,'-MD','-MF',str(dep),'-c',str(src),'-o',str(obj)],f'compile-{j}');dd=deps(dep.read_text());fresh=deps(run([cmd[0],*flags,'-M',str(src)],f'fresh-M-{j}'));assert dd==fresh
  policies=[v for v in dd if pathlib.Path(v).name in ['glm_model_resident_policy_r5640.hpp','glm_rows32_policy_r5731.hpp']];assert policies and all(v.startswith(str(b)+'/tree/')for v in policies)
  replaced[old]=str(obj);compiled.append({'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'replaces':old,'MD_SHA256':sha(dep),'policy_dependencies_SHA256':{p:sha(pathlib.Path(p))for p in policies}})
 link=[replaced.get(v,v)for v in link];lib=b/'libglm_PP2_B32_r8843.so';link[link.index('-o')+1]=str(lib);run(link,'link');symbols=run(['nm','-D','--defined-only',str(lib)],'exports');assert all(v in symbols for v in ['modern_glm_model_create','modern_glm_model_resident_mask_r5640','modern_glm_vector_policy_r8540','modern_glm_prefill_prefix_plan_counts_r8703']);assert 'modern_glm_tile2_policy_r8830'not in symbols and 'modern_glm_quant_reuse_policy_r8832'not in symbols
 assert all(sha(o)==h for o,h in objects.items());q.update(completed=True,compiled_TUs=6,compiled=compiled,library=str(lib),library_SHA256=sha(lib),reused_objects_SHA256={v:h for v,h in objects.items()if v not in replaced},original_private_headers_SHA256=headers,source_manifest_SHA256=sha(mp),scope='PP2 constructor32/active32 optional, M8 cohort<=8 unchanged; current522prefix/vector/SDK/math; allnewkerneloptsabsent. Commonbank B16->B32 Model NOT_RUN.')
except BaseException as e:q['error']=repr(e)
finally:q['seconds']=time.monotonic()-begin;(b/'CPU-build.json').write_text(json.dumps(q,indent=2))
print(json.dumps({k:q.get(k)for k in ['completed','producer','seconds','library_SHA256','error']}));assert q['completed']
