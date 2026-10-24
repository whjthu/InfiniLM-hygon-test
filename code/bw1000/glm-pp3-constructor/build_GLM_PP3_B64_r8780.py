import pathlib,json,hashlib,subprocess,shutil,shlex,os,time
r=pathlib.Path('/refactor');b=r/'candidates/glm-PP3-B64-r8780';assert not b.exists();b.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();begin=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[],'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
def run(a,n):
 q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=240);(b/(n+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError((n,p.returncode,p.stderr[-5000:]))
 return p.stdout
def mirror(p):return b/'tree'/pathlib.Path(p).relative_to(r)
try:
 m=json.loads((r/'incoming/GLM-PP3-build-input-r8780.json').read_text());cpu=json.loads((r/'candidates/glm-prefix-bridge-r8705/CPU-build.json').read_text());assert cpu['completed']and sha(cpu['library'])==cpu['library_SHA256']=='5f3dd3d9c2a6715992be48f5de514a145c11d971cb7e07a4ec1bc10fb6a7ad52'
 link=next(list(c)for c in cpu['commands']if '-shared'in c);objects={v:sha(v)for v in link if v.endswith('.o')};assert len(objects)==28
 affected=[o for o in m['objects']if o.get('resident_headers')];assert len(affected)==6
 overrides=m['overrides'];original_headers={}
 for rec in affected:
  assert sha(rec['object'])==rec['SHA256'];dep=pathlib.Path(rec['object']).with_suffix('.d');vals=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])
  for v in vals:
   p=pathlib.Path(v)
   if v.startswith('/refactor/candidates/')and '/src/'in v and p.suffix in ['.hpp','.h']:
    dst=mirror(p);dst.parent.mkdir(parents=True,exist_ok=True);original_headers[v]=sha(p)
    if p.name in overrides:
     ov=overrides[p.name];src=r/'incoming'/ov['incoming'];assert sha(src)==ov['SHA256'];shutil.copyfile(src,dst)
    else:shutil.copyfile(p,dst)
  src=pathlib.Path(rec['source'][0]);ov=overrides[src.name];incoming=r/'incoming'/ov['incoming'];assert sha(incoming)==ov['SHA256'];dst=mirror(src);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(incoming,dst)
 allcmd=[]
 for folder in sorted(set(str(pathlib.Path(rec['object']).parent)for rec in affected)):
  p=json.loads((pathlib.Path(folder)/'CPU-build.json').read_text());allcmd.extend(c for c in p['commands']if isinstance(c,list))
 replaced={};compiled=[]
 for i,rec in enumerate(affected):
  old=rec['object'];cmd=next(c for c in allcmd if '-c'in c and '-o'in c and c[c.index('-o')+1]==old);src=mirror(pathlib.Path(rec['source'][0]));flags=[];j=1
  while j<len(cmd):
   if cmd[j]in ['-c','-o','-MF','-MT','-MQ']:j+=2;continue
   if cmd[j]in ['-MD','-MMD']:j+=1;continue
   v=cmd[j]
   if v.startswith('-I/refactor/candidates/')and '/src'in v:v='-I'+str(mirror(pathlib.Path(v[2:])))
   flags.append(v);j+=1
  obj=b/f'private-{i}.o';dep=b/f'private-{i}.d';run([cmd[0],*flags,'-MD','-MF',str(dep),'-c',str(src),'-o',str(obj)],f'compile-{i}')
  deps=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]);policies=[v for v in deps if pathlib.Path(v).name in ['glm_model_resident_policy_r5640.hpp','glm_rows16_policy_r5702.hpp']];assert policies and all(v.startswith(str(b)+'/tree/')for v in policies)
  replaced[old]=str(obj);compiled.append({'source':str(src),'SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'replaces':old,'MD_SHA256':sha(dep),'policy_dependencies':policies})
 link=[replaced.get(v,v)for v in link];lib=b/'libglm_PP3_B64_r8780.so';link[link.index('-o')+1]=str(lib);run(link,'link');symbols=run(['nm','-D','--defined-only',str(lib)],'exports');assert all(v in symbols for v in ['modern_glm_model_create','modern_glm_model_resident_mask_r5640','modern_glm_vector_policy_r8540','modern_glm_prefill_prefix_plan_counts_r8703'])
 for ob,h in objects.items():assert sha(ob)==h
 q.update(completed=True,compiled=compiled,library=str(lib),library_SHA256=sha(lib),reused_objects_SHA256={v:h for v,h in objects.items()if v not in replaced},original_private_headers_SHA256=original_headers,scope='6 dependent private hostTUs+2 headers; noSDK/GPU compile, math/kernel unchanged; PP3 B64M8 only; unvalidatedGPU scope/performance/peak')
except BaseException as e:q['error']=repr(e);raise
finally:q['seconds']=time.monotonic()-begin;(b/'CPU-build.json').write_text(json.dumps(q,indent=2))
print(json.dumps({k:q.get(k)for k in ['completed','producer','seconds','library_SHA256','error']}))
