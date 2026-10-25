import pathlib,json,hashlib,subprocess,re,os,time,shutil
r=pathlib.Path('/refactor');i=r/'incoming';b=r/'candidates/glm-positions-r9022';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();old=json.loads((b/'CPU-build.json').read_text());assert not old['completed'];assert not(b/'CPU-build-first-error.json').exists();shutil.copyfile(b/'CPU-build.json',b/'CPU-build-first-error.json');closed=b/'closed';closed.mkdir();q={'completed':False,'GPU':False,'dlopen':False,'GPU_TUs_compiled_on_CPU':0,'HOST_TUs_compiled':0,'reused_prepared_HOST_TUs':1,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'first_error_handle':old['producer'],'mechanical_repair_attempts':1,'commands':[]};start=time.perf_counter()
def run(a):
 q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=100);(b/f'repair-command-{len(q["commands"])}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[:18000]
try:
 doc=json.loads((r/'candidates/glm-PP2-B32-r8843/CPU-build.json').read_text());oldcache='/refactor/candidates/glm-PP2-B32-r8843/private-5.o';a=next(list(a)for a in doc['commands']if '-c'in a and '-o'in a and a[a.index('-o')+1]==oldcache);md=pathlib.Path(a[a.index('-MF')+1]);deps=md.read_text().replace('\\\n',' ').split()
 private=['glm_cache_positions_r9022.cpp','glm_cache_positions_header_r9022.hpp','glm_native_positions_header_r9021.hpp','glm_prepare_positions_header_r9021.hpp'];bindings={}
 for name in private:
  s=(i/name).read_text()
  def replace(m):
   n=m[1]
   if n in private:return '#include "'+n+'"'
   if n.startswith('/'):return m[0]
   hits=sorted(set(x for x in deps if x.endswith('/'+n)))
   assert len(hits)==1,(name,n,hits)
   bindings[n]=hits[0];return '#include "'+hits[0]+'"'
  s=re.sub(r'#include\s+"([^"]+)"',replace,s);(closed/name).write_text(s)
 po=r/'candidates/glm-positions-r9021/glm_prepare_positions_r9021.cpp.o';assert sha(i/'glm_prepare_positions_r9021.cpp')=='19058b777f62ae825c58d09dff9ae0dbbdffe1144952c420a50b0d05efd34e58';ph=sha(po);obj=b/'cache.o';a[a.index('-c')+1]=str(closed/private[0]);a[a.index('-o')+1]=str(obj);a[a.index('-MF')+1]=str(b/'cache-repaired.d');run(a);q['HOST_TUs_compiled']=1
 parent=json.loads((r/'candidates/glm-exact-budget-r8854/CPU-build.json').read_text());link=next(list(x)for x in parent['commands']if '-shared'in x);objects={x:sha(x)for x in link if x.endswith('.o')};oldprepare='/refactor/candidates/glm-transient-layer-r5681/private-8.o';assert objects[oldcache]=='290d8c38d7aa7afc50ccd1deab97bed2e56906aed0489a17273561bd72925dea'and objects[oldprepare]=='d381e4ee8632de65813b9c3507ecfebe929bc1bac6890a34ea0276b2cc0a6809';lib=b/'libglm_positions_r9022.so';link=[str(po)if x==oldprepare else str(obj)if x==oldcache else x for x in link];link[link.index('-o')+1]=str(lib);run(link);run(['nm','-D','--defined-only',str(lib)]);assert all(sha(p)==h for p,h in objects.items())and sha(po)==ph;q.update(completed=True,library=str(lib),library_SHA256=sha(lib),reused_prepare_object=str(po),reused_prepare_object_SHA256=ph,reused_objects_SHA256={p:h for p,h in objects.items()if p not in [oldprepare,oldcache]},bindings=bindings,closed_sources_SHA256={p.name:sha(p)for p in closed.iterdir()},elapsed_seconds=time.perf_counter()-start)
except BaseException as exc:q['failure']=repr(exc)
(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:v for k,v in q.items()if k not in ['commands','reused_objects_SHA256','bindings']}));assert q['completed']
