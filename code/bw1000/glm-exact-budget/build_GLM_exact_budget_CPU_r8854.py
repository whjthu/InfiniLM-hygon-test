import pathlib,json,hashlib,subprocess,shutil,os,time,shlex
r=pathlib.Path('/refactor');b=r/'candidates/glm-exact-budget-r8854';assert not b.exists();b.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]}
def run(cmd,label):
 q['commands'].append(cmd);p=subprocess.run(cmd,capture_output=True,text=True,timeout=200);(b/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.stderr[-5000:]);return p.stdout
try:
 mp=r/'incoming/GLM-exact-budget-CPU-manifest-r8854.json';m=json.loads(mp.read_text())
 for n,h in m['sources_SHA256'].items():assert sha(r/'incoming'/n)==h,n
 cp=r/'candidates/glm-resident-exact-r8852/CPU-build.json';c=json.loads(cp.read_text());assert c['completed']and sha(c['library'])==c['library_SHA256']==m['parent_DSO_SHA256']
 prior=json.loads((r/'candidates/glm-PP2-B32-r8843/CPU-build.json').read_text());rec=next(x for x in prior['compiled']if pathlib.Path(x['source']).name=='modern_glm_cached_model_bridge_r809.cpp');old=rec['object'];assert c['reused_objects_SHA256'][old]==sha(old)==rec['object_SHA256']
 cmd=next(x for x in prior['commands']if '-c'in x and '-o'in x and x[x.index('-o')+1]==old);dep=pathlib.Path(old).with_suffix('.d');headers={}
 for v in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):
  p=pathlib.Path(v)
  if p.suffix in('.hpp','.h')and v.startswith('/refactor/candidates/'):
   dst=b/'src'/p.name;dst.parent.mkdir(exist_ok=True);h=sha(p)
   if dst.exists():assert sha(dst)==h,('headerbasenamecollision',p)
   else:shutil.copyfile(p,dst)
   headers[v]=h
 source=b/'src/modern_glm_cached_model_bridge_r809.cpp';shutil.copyfile(r/'incoming/GLM_exact_budget_bridge_r8854.cpp',source);shutil.copyfile(r/'incoming/glm_resident_exact_budget_r8854.hpp',source.parent/'glm_resident_exact_budget_r8854.hpp')
 flags=[];i=1
 while i<len(cmd):
  if cmd[i]in('-c','-o','-MF','-MT','-MQ'):i+=2;continue
  if cmd[i]in('-MD','-MMD'):i+=1;continue
  flags.append(cmd[i]);i+=1
 obj=b/'bridge.o';md=b/'bridge.d';run([cmd[0],'-I'+str(source.parent),*flags,'-MD','-MF',str(md),'-c',str(source),'-o',str(obj)],'compile')
 check=b/'check.cpp';shutil.copyfile(r/'incoming/check_GLM_exact_budget_r8854.cpp',check);exe=b/'check';run([cmd[0],'-std=c++17','-I'+str(source.parent),str(check),'-o',str(exe)],'checkbuild');budget=json.loads(run([str(exe)],'checkrun'));assert budget['resident_bank_bytes']==160*1024**2
 assert 'glm_resident_exact_budget_r8854.hpp'in md.read_text()
 link=next(list(x)for x in c['commands']if '-shared'in x);original={x:sha(x)for x in link if x.endswith('.o')};assert len(original)==28 and old in original
 link=[str(obj)if x==old else x for x in link];lib=b/'libglm_exact_budget_r8854.so';link[link.index('-o')+1]=str(lib);run(link,'link')
 symbols=run(['nm','-D','--defined-only',str(lib)],'exports');oldsymbols=run(['nm','-D','--defined-only',c['library']],'oldexports');required={x.split()[-1]for x in oldsymbols.splitlines()if x.split()and(x.split()[-1].startswith('modern_glm_')or x.split()[-1].startswith('glm_'))};assert required<={x.split()[-1]for x in symbols.splitlines()if x.split()}
 assert all(sha(p)==h for p,h in original.items())
 q.update(completed=True,compiled_TUs=1,compiled=[{'source':str(source),'source_SHA256':sha(source),'object':str(obj),'object_SHA256':sha(obj),'replaces':old,'MD_SHA256':sha(md),'exact_budget_header_SHA256':sha(source.parent/'glm_resident_exact_budget_r8854.hpp')}],reused_objects_SHA256={p:h for p,h in original.items()if p!=old},original_headers_SHA256=headers,library=str(lib),library_SHA256=sha(lib),preserved_CAPI_symbols=sorted(required),resident_exact_budget=budget,source_manifest_SHA256=sha(mp),scope=m['scope'])
except BaseException as e:q['error']=repr(e)
finally:q['seconds']=time.monotonic()-start;(b/'CPU-build.json').write_text(json.dumps(q,indent=2))
print(json.dumps({k:q.get(k)for k in['completed','producer','seconds','library_SHA256','resident_exact_budget','error']}));assert q['completed']
