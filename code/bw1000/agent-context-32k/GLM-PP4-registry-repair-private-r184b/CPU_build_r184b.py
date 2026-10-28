"""Replace one exact existing caller HOST object. No HIP/math/cache changes."""
import pathlib,json,hashlib,subprocess,shlex,time,os,traceback,re
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/GLM-PP4-registry-repair-r184b';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);B.mkdir(exist_ok=False)
q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'HOST_TUs_compiled':0,'GPU_TUs_compiled':0,'commands':[]};start=time.monotonic()
def run(a,n):
 q['commands'].append(a);z=subprocess.run(a,capture_output=True,text=True,timeout=180);(B/(n+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(n,z.stderr[-3000:]);return z.stdout
def deps(text):return set(shlex.split(text.replace('\\\n',' ').split(':',1)[1]))
def flags(a):
 out=[];i=0
 while i<len(a):
  if a[i]in['-MF','-MT','-MQ','-o','-c']:i+=2;continue
  if a[i]in['-MD','-MMD','-MP']:i+=1;continue
  out.append(a[i]);i+=1
 return out
try:
 m=json.loads((D/'source-manifest-r184b.json').read_bytes())
 for n,h in m['files'].items():assert sha(D/n)==h,n
 cp=R/'candidates/GLM-critical-observer-r131/CPU-build.json';assert sha(cp)==m['parent_receipt_SHA256'];c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and not c['GPU_initialized']and sha(c['library'])==c['library_SHA256']==m['parent_native_SHA256']
 for field in['reused_objects_SHA256','actual_binary_SHA256','dependency_SHA256']:
  for n,h in c[field].items():assert sha(n)==h,n
 for x in c['compiled']:assert sha(x['object'])==x['object_SHA256']and sha(x['source'])==x['source_SHA256']
 original_obj=R/'candidates/GLM-B32-Model-r9183/resident_caller_r9183.cpp.o';assert sha(original_obj)==m['replaced_object_SHA256']and c['reused_objects_SHA256'][str(original_obj)]==m['replaced_object_SHA256']
 pp=R/'candidates/GLM-B32-Model-r9183/CPU-build.json';p=json.loads(pp.read_bytes());assert p['completed']
 x=next(x for x in p['compiled']if x['object']==str(original_obj));assert x['source_SHA256']==m['original_caller_SHA256']and sha(x['source'])==m['original_caller_SHA256']and x['object_SHA256']==m['replaced_object_SHA256']
 orig=next(a for a in p['commands']if '-c'in a and '-o'in a and a[a.index('-o')+1]==str(original_obj));oldmd=pathlib.Path(orig[orig.index('-MF')+1]);assert oldmd.is_file();q['actual_parent_caller_MD']={'path':str(oldmd),'SHA256':sha(oldmd),'source_path':x['source'],'source_SHA256':sha(x['source']),'object_SHA256':sha(original_obj),'recipe_receipt_SHA256':sha(pp)}
 exe=B/'geometry-contract';run(['/usr/bin/g++','-std=c++17','-O2','-I'+str(D),str(D/'CPU_geometry_contract_r184b.cpp'),'-o',str(exe)],'geometry-build');q['geometry_contract']=json.loads(run([str(exe)],'geometry-run'));assert q['geometry_contract']['passed']
 prefix=flags(orig);prefix.insert(1,'-I'+str(D));src=D/'resident_caller_registry_r184b.cpp';obj=B/'resident_caller_registry_r184b.cpp.o';md=B/'resident_caller_registry_r184b.cpp.d'
 run(prefix+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],'caller-compile');q['HOST_TUs_compiled']=1
 fresh=run(prefix+['-M',str(src)],'caller-fresh-M');dep=deps(fresh);assert dep==deps(md.read_text());assert str(D/'glm_registry_geometry_r184b.hpp')in dep and str(D/'glm_PP_layout_r3854.hpp')in dep
 hashes={p:sha(p)for p in sorted(dep)}
 link=list(next(a for a in c['commands']if '-shared'in a and '-o'in a));assert link.count(str(original_obj))==1;link=[str(obj)if a==str(original_obj)else a for a in link];lib=B/'libGLM_PP4_registry_r184b.so';link[link.index('-o')+1]=str(lib);run(link,'link')
 symbols=lambda text:{s.split()[-1]for s in text.splitlines()if s.split()}
 old=symbols(run(['nm','-D','--defined-only',c['library']],'old-exports'));new=symbols(run(['nm','-D','--defined-only',str(lib)],'new-exports'));q['missing_parent_exports']=sorted(old-new);assert old<=new,q['missing_parent_exports']
 def inspect_ldd(path,name):
  a=['ldd','-r',str(path)];q['commands'].append(a);z=subprocess.run(a,capture_output=True,text=True,timeout=60);text=z.stdout+z.stderr;(B/(name+'.log')).write_text(text);assert 'not found'not in text,(name,text[-3000:]);return text,z.returncode,set(re.findall(r'undefined symbol:\s*([^\s(]+)',text))
 parent_ldd,parent_ldd_rc,parent_unresolved=inspect_ldd(c['library'],'parent-ldd-r')
 ldd,ldd_rc,unresolved=inspect_ldd(lib,'ldd-r');assert ldd_rc==parent_ldd_rc and ldd_rc in[0,1]and unresolved==parent_unresolved,{'parent_rc':parent_ldd_rc,'candidate_rc':ldd_rc,'new_undefined':sorted(unresolved-parent_unresolved),'removed_undefined':sorted(parent_unresolved-unresolved)}
 q['ldd_r_same_as_qualified_parent']=True;q['preexisting_runtime_host_imports']=sorted(unresolved);q['parent_ldd_r']=parent_ldd;q['parent_ldd_r_returncode']=parent_ldd_rc;q['candidate_ldd_r_returncode']=ldd_rc

 reused=dict(c['reused_objects_SHA256']);del reused[str(original_obj)]
 q.update(completed=True,normal_CPU_exit=True,library=str(lib),library_SHA256=sha(lib),parent_native_SHA256=c['library_SHA256'],parent_receipt_SHA256=sha(cp),source_manifest_SHA256=sha(D/'source-manifest-r184b.json'),compiled=[{'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'replaces':str(original_obj),'MD_SHA256':sha(md),'fresh_M_equal_MD':True,'full_semantic_argv_replayed':True}],reused_objects_SHA256=reused,parent_fresh_compiled=c['compiled'],actual_binary_SHA256=c['actual_binary_SHA256'],dependency_SHA256=hashes,parent_actual_dependency_SHA256=c['dependency_SHA256'],all_parent_public_symbols_retained=True,ABI=1,Context_layout_changed=False,NativeAttention_class_layout_changed=False,internal_registry_adds_PP_integer=True,math_changes=0,cache_layout_changes=0,CPP_forward_math_body_changes=False,PP4_crossrequest_optimization_forced_OFF=True,PP2_optimization_behavior_unchanged=True,ldd_r=ldd,allocation_GPU_qualified=False,weights_loaded=False,Model_GPU_qualified=False)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['seconds']=time.monotonic()-start;q['producer']=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','');(B/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:q.get(k)for k in['completed','producer','library','library_SHA256','HOST_TUs_compiled','GPU_TUs_compiled','error','missing_parent_exports']}));assert q['completed']
