"""Two private TUs, reuse all other actual rows16 objects/SDK. CPU only."""
import pathlib,json,hashlib,subprocess,shutil,shlex,time,os
r=pathlib.Path('/refactor');i=r/'incoming';base=r/'candidates/glm-crossrequest-model-r7328';assert not base.exists();(base/'src').mkdir(parents=True)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((i/'GLM-crossrequest-Model-CPU-manifest-r7328.json').read_text());parent=json.loads((r/'candidates/glm-rows16-model-r5707/CPU-build.json').read_text());layer=json.loads((r/'candidates/glm-transient-layer-r5681/CPU-build.json').read_text())
assert sha(parent['library'])==parent['library_SHA256']==m['parent_DSO_SHA256'];assert sha(r/'candidates/glm-transient-layer-r5681/CPU-build.json')==m['layer_receipt_SHA256']
for p,h in parent['reused_objects_SHA256'].items():assert sha(p)==h
for p in parent['compiled']:assert sha(p['object'])==p['object_SHA256']
for p,h in parent['Runtime_and_extension_SHA256'].items():assert sha(p)==h
for n,h in parent['SDK_libraries_SHA256'].items():assert sha(r/'install/hygon-glm-5f91ad7-r3672/lib'/n)==h
commands=[];report={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True};start=time.perf_counter()
def run(a,label):
 commands.append(a);p=subprocess.run(a,text=True,capture_output=True);(base/(label+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError((label,p.returncode,p.stderr[-7000:]))
 return p.stdout
parse=lambda s:set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]))
try:
 targets=[('glm_crossrequest_Model_cache_view_r7328.cpp','glm_moe_dsa_projected_cache_view_r927.cpp','cache.o'),('glm_crossrequest_Model_helper_r7328.cpp','glm_moe_dsa_dense_modes_r48.cpp','helper.o')];built=[];deps_all={};replacement={}
 for name,oldname,objectname in targets:
  src=base/'src'/name;assert sha(i/name)==m['sources_SHA256'][name];shutil.copyfile(i/name,src)
  rec=next(v for v in layer['compiled']if pathlib.Path(v['source']).name==oldname);assert sha(rec['source'])==rec['source_SHA256']and sha(rec['object'])==rec['object_SHA256']
  a=next(a for a in layer['commands']if'-c'in a and rec['source']in a);flags=[];j=1
  while j<len(a):
   if a[j]in['-c','-o','-MF','-MT','-MQ']:j+=2;continue
   if a[j]in['-MD','-MMD']or a[j]=='-I/refactor/incoming':j+=1;continue
   flags.append(a[j]);j+=1
  obj=base/objectname;dep=obj.with_suffix('.d');run([a[0],*flags,'-MD','-MF',str(dep),'-c',str(src),'-o',str(obj)],'compile-'+objectname)
  ds=parse(dep.read_text());assert ds==parse(run([a[0],*flags,'-M',str(src)],'fresh-M-'+objectname));assert not any('/incoming/'in d for d in ds)
  known=layer.get('dependency_SHA256',{})
  critical={d:sha(d)for d in ds if any(s in d for s in ['/csrc/runtime/','/csrc/execution/','/src/glm_','/install/hygon-glm-'])}
  for p,h in critical.items():
   if p in known:assert known[p]==h,('changed_actual_dependency',p)
  for p in ds:deps_all[p]=sha(p)
  built.append({'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'MD_SHA256':sha(dep),'critical_headers':critical,'dependency_count':len(ds),'original_template_source':rec['source']})
  if objectname=='cache.o':replacement[rec['object']]=str(obj)
 link=next(list(a)for a in parent['commands']if'-shared'in a);objects=[a for a in link if a.endswith('.o')];assert len(objects)==26 and len(replacement)==1 and all(link.count(k)==1 for k in replacement)
 link=[replacement.get(a,a)for a in link];link.insert(1,str(base/'helper.o'));lib=base/'libglm_crossrequest_model_r7328.so';link[link.index('-o')+1]=str(lib);run(link,'link')
 exports=run(['nm','-D','--defined-only',str(lib)],'exports');old=run(['nm','-D','--defined-only',parent['library']],'parent-exports');names={l.split()[-1]for l in exports.splitlines()};api={l.split()[-1]for l in old.splitlines()if l.split()[-1].startswith(('modern_glm_','glm_layer_'))};assert api<=names;assert 'modern_glm_crossrequest_prefill_calls_r7328'in names
 dynamic=run(['readelf','-d',str(lib)],'dynamic');assert 'libglm_rows16_model_r5707.so'not in dynamic
 undefined=run(['nm','-D','--undefined-only',str(lib)],'undefined');old_undefined=run(['nm','-D','--undefined-only',parent['library']],'parent-undefined')
 report.update(completed=True,elapsed_seconds=time.perf_counter()-start,parent_DSO_SHA256=parent['library_SHA256'],library=str(lib),library_SHA256=sha(lib),compiled=built,compiled_TUs=2,reused_objects_SHA256={p:sha(p)for p in objects if p not in replacement},object_count=27,dependency_SHA256=deps_all,fresh_M_equal_MD=True,critical_header_policy='Existing actual headers only, no private class/header edits, no incoming header shadow; actual known historical hashes checked where recorded.',SDK_libraries_SHA256=parent['SDK_libraries_SHA256'],Runtime_and_extension_SHA256=parent['Runtime_and_extension_SHA256'],native_SHA256=parent['native_SHA256'],exports=exports,undefined=undefined,parent_undefined=old_undefined,dynamic=dynamic,parent_API_retained=len(api),commands=commands,scope='Only uniform8requests512past0 fullcausalprefill optional route. DefaultOFF, original cache writes/hostcommit/read and lengthpublication kept. Consumer must lock actual mode2. NoGPU F64/newweights/SDK update.',NOT_RUN=['CPUdlopen','GPU','ModelIDs','Modelperformance'])
except BaseException as e:
 report.update(error=repr(e),commands=commands);(base/'CPU-failure.json').write_text(json.dumps(report,indent=2));raise
else:
 (base/'CPU-build.json').write_text(json.dumps(report,indent=2));(r/'results/glm-crossrequest-model-CPU-r7328.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k]for k in['completed','producer','elapsed_seconds','library_SHA256','compiled_TUs','object_count']}))
