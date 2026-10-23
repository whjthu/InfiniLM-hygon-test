"""Only two private Model TUs; actual DTK CPU compile/link, no dlopen/GPU."""
import pathlib,json,hashlib,subprocess,shutil,shlex,time,datetime,os
r=pathlib.Path('/refactor');i=r/'incoming';base=r/'candidates/glm-rows16-model-r5707';mp=i/'GLM-ROWS16-CPU-manifest-r5707.json';m=json.loads(mp.read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for n,h in m['sources_SHA256'].items():assert sha(i/n)==h,n
assert not base.exists();(base/'src').mkdir(parents=True);commands=[];start=time.perf_counter();report={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
def run(a,label):
 commands.append(a);p=subprocess.run(a,text=True,capture_output=True);(base/(label+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError((label,p.returncode,p.stderr[-7000:]))
 return p.stdout
def deps(s):return set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]))
try:
 docs={}
 for rel,h in m['receipt_SHA256'].items():assert sha(r/rel)==h,rel;docs[rel]=json.loads((r/rel).read_text())
 parent=docs['candidates/glm-vector-model-r8542/CPU-build.json'];assert parent['completed']and parent['ABI']==1 and parent['fresh_M_equal_MD']and sha(parent['library'])==parent['library_SHA256']==m['parent_DSO_SHA256']
 for path,h in parent['reused_objects_SHA256'].items():assert sha(path)==h,path
 for c in parent['compiled']:assert sha(c['object'])==c['object_SHA256']and sha(c['source'])==c['source_SHA256']
 for path,h in parent['dependency_SHA256'].items():assert sha(path)==h,path
 for path,h in parent['Runtime_and_extension_SHA256'].items():assert sha(path)==h,path
 for n,h in parent['SDK_libraries_SHA256'].items():assert sha(r/'install/hygon-glm-5f91ad7-r3672/lib'/n)==h,n
 assert sha(r/'candidates/glm-hybrid-decode-r5386/libnative.so')==parent['native_SHA256']==m['native_SHA256']
 inputs=json.loads((i/'r5706-GLM-rows16-link-inputs.json').read_text());assert inputs['receipt_SHA256']==m['receipt_SHA256']and inputs['current_library_SHA256']==m['parent_DSO_SHA256']
 for target in inputs['target_TUs']:
  assert sha(target['source'])==target['source_SHA256']and sha(target['object'])==target['object_SHA256']and sha(target['MD_file'])==target['MD_SHA256']
  assert inputs['actual_MD_reverse_source_dependencies'][target['source']]==[target['source']]
 for n,alias in m['source_files'].items():shutil.copyfile(i/alias,base/'src'/n)
 # Existing public class layouts and inline Slots/policy remain byte-identical.
 for p,h in inputs['public_class_headers'].items():assert sha(p)==h,p
 replacements={};compiled=[];all_deps={};compiler=shutil.which('g++')
 for j,target in enumerate(inputs['target_TUs']):
  name=pathlib.Path(target['source']).name;src=base/'src'/name;obj=base/f'rows16-{j}.o';dep=base/f'rows16-{j}.d';flags=['-I'+str(base/'src'),*target['flags']];assert '-D_GLIBCXX_USE_CXX11_ABI=1'in flags
  run([compiler,*flags,'-MD','-MF',str(dep),'-c',str(src),'-o',str(obj)],f'compile-{j}');ds=deps(run([compiler,*flags,'-M',str(src)],f'fresh-M-{j}'));assert ds==deps(dep.read_text())
  for p in ds:all_deps[p]=sha(p)
  compiled.append({'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'MD_file':str(dep),'MD_SHA256':sha(dep),'dependency_count':len(ds)})
  replacements[target['object']]=str(obj)
 args=next(list(a)for a in parent['commands']if '-shared'in a)
 assert all(args.count(a)==1 for a in replacements);args=[replacements.get(a,a)for a in args];args[0]=compiler;lib=base/'libglm_rows16_model_r5707.so';args[args.index('-o')+1]=str(lib);run(args,'link')
 exports=run(['nm','-D','--defined-only',str(lib)],'exports');old=run(['nm','-D','--defined-only',parent['library']],'parent-exports');names=[a.split()[-1]for a in exports.splitlines()];oldnames={a.split()[-1]for a in old.splitlines()}
 api={n for n in oldnames if n.startswith('modern_glm_')or n.startswith('glm_grouped_')or n.startswith('glm_vec_')};assert api<={*names};assert all(names.count(n)==1 for n in api)
 dyn=run(['readelf','-d',str(lib)],'dynamic');assert 'libglm_vector_model_r8542'not in dyn and 'libglm_transient_model_r5690'not in dyn
 report.update(completed=True,ABI=1,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,source_manifest_SHA256=sha(mp),parent_DSO_SHA256=parent['library_SHA256'],parent_receipt_SHA256=m['receipt_SHA256'],compiled_TUs=2,compiled=compiled,dependency_SHA256=all_deps,fresh_M_equal_MD=True,public_headers_unchanged=inputs['public_class_headers'],reverse_source_dependencies=inputs['actual_MD_reverse_source_dependencies'],library=str(lib),library_SHA256=sha(lib),native_SHA256=parent['native_SHA256'],SDK_libraries_SHA256=parent['SDK_libraries_SHA256'],Runtime_and_extension_SHA256=parent['Runtime_and_extension_SHA256'],reused_objects_SHA256={a:sha(a)for a in args if a.endswith('.o')and a not in replacements.values()},ELF_dynamic=dyn,parent_API_export_count=len(api),commands=commands,scope='DefaultunchangedROWS64; optionalROWS16resident geometry/mask and128MiB/layer preadmit. Same originalModelRuntime/math. CPUonly no dlopen/GPU/weights.',NOT_RUN=['CPUdlopen','GPUABI','ModelIDs','Modelperformance'])
except BaseException as exc:
 report.update(failure=repr(exc),commands=commands);(base/'CPU-failure.json').write_text(json.dumps(report,indent=2));raise
else:
 for p in[base/'CPU-build.json',r/'results/glm-rows16-CPU-r5707.json']:assert not p.exists();p.write_text(json.dumps(report,indent=2))
 print(json.dumps({k:report[k]for k in['completed','producer','compiled_TUs','elapsed_seconds','library_SHA256','parent_API_export_count']}))
