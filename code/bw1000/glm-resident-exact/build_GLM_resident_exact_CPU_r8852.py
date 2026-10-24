import pathlib,json,hashlib,subprocess,shutil,os,time,shlex
r=pathlib.Path('/refactor');b=r/'candidates/glm-resident-exact-r8852';assert not b.exists();b.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]}
def run(cmd,label):
 q['commands'].append(cmd);p=subprocess.run(cmd,capture_output=True,text=True,timeout=200);(b/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.stderr[-5000:]);return p.stdout
try:
 mp=r/'incoming/GLM-resident-exact-CPU-manifest-r8852.json';m=json.loads(mp.read_text())
 for n,h in m['sources_SHA256'].items():assert sha(r/'incoming'/n)==h,n
 cp=r/'candidates/glm-PP2-B32-r8843/CPU-build.json';c=json.loads(cp.read_text());assert c['completed']and sha(c['library'])==c['library_SHA256']==m['parent_DSO_SHA256']
 rec=next(x for x in c['compiled']if pathlib.Path(x['source']).name=='glm_model_resident_methods_r5640.cpp');old=rec['object'];cmd=next(x for x in c['commands']if '-c'in x and '-o'in x and x[x.index('-o')+1]==old);dep=pathlib.Path(old).with_suffix('.d');headers={}
 for v in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):
  p=pathlib.Path(v)
  if p.suffix in('.hpp','.h')and v.startswith('/refactor/candidates/'):
   dst=b/'src'/p.name;dst.parent.mkdir(exist_ok=True);h=sha(p)
   if dst.exists():assert sha(dst)==h,('headerbasenamecollision',p)
   else:shutil.copyfile(p,dst)
   headers[v]=h
 source=b/'src/glm_model_resident_methods_r5640.cpp';shutil.copyfile(r/'incoming/GLM_resident_exact_methods_r8852.cpp',source)
 flags=[];i=1
 while i<len(cmd):
  if cmd[i]in('-c','-o','-MF','-MT','-MQ'):i+=2;continue
  if cmd[i]in('-MD','-MMD'):i+=1;continue
  flags.append(cmd[i]);i+=1
 obj=b/'resident.o';md=b/'resident.d';run([cmd[0],'-I'+str(source.parent),*flags,'-MD','-MF',str(md),'-c',str(source),'-o',str(obj)],'compile')
 link=next(list(x)for x in c['commands']if '-shared'in x);original={x:sha(x)for x in link if x.endswith('.o')};assert len(original)==28 and old in original
 link=[str(obj)if x==old else x for x in link];lib=b/'libglm_resident_exact_r8852.so';link[link.index('-o')+1]=str(lib);run(link,'link');symbols=run(['nm','-D','--defined-only',str(lib)],'exports')
 required=['modern_glm_model_create','modern_glm_model_destroy','modern_glm_model_shutdown','modern_glm_model_forward','modern_glm_model_resident_mask_r5640','modern_glm_model_resident_report_r5640','modern_glm_vector_policy_r8540','modern_glm_prefill_prefix_plan_counts_r8703']
 assert all(x in symbols for x in required);assert all(sha(p)==h for p,h in original.items());assert 'modern_glm_tile2_policy_r8830'not in symbols and 'modern_glm_quant_reuse_policy_r8832'not in symbols
 q.update(completed=True,compiled_TUs=1,compiled=[{'source':str(source),'source_SHA256':sha(source),'object':str(obj),'object_SHA256':sha(obj),'replaces':old,'MD_SHA256':sha(md)}],reused_objects_SHA256={p:h for p,h in original.items()if p!=old},original_headers_SHA256=headers,library=str(lib),library_SHA256=sha(lib),required_actual_consumer_symbols=required,source_manifest_SHA256=sha(mp),scope='OnlyresidentK/V empty_parameter+allocationmetadata;current198rows32/prefix/vector/kernel/math/SDK retained;NoGPUhit/Modelfit proof.')
except BaseException as e:q['error']=repr(e)
finally:q['seconds']=time.monotonic()-start;(b/'CPU-build.json').write_text(json.dumps(q,indent=2))
print(json.dumps({k:q.get(k)for k in['completed','producer','seconds','library_SHA256','error']}));assert q['completed']
