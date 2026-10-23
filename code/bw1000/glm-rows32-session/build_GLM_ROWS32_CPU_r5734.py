"""Private actual-DTK six-TU CPU closure. No dlopen, CUDA call or Model IO."""
import pathlib,json,hashlib,subprocess,shutil,shlex,time,datetime,os
r=pathlib.Path('/refactor');i=r/'incoming';base=r/'candidates/glm-rows32-model-r5734';mp=i/'GLM-ROWS32-CPU-manifest-r5734.json';m=json.loads(mp.read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for n,h in m['sources_SHA256'].items():assert sha(i/n)==h,n
assert not base.exists();(base/'src').mkdir(parents=True);commands=[];start=time.perf_counter();report={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
def run(a,label):
 commands.append(a);p=subprocess.run(a,text=True,capture_output=True);(base/(label+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError((label,p.returncode,p.stderr[-7000:]))
 return p.stdout
def deps(s):return set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]))
try:
 docs={};records={};owners={}
 for rel,h in m['receipt_SHA256'].items():
  assert sha(r/rel)==h,rel;q=json.loads((r/rel).read_text());assert q['completed'];docs[rel]=q
  for c in q.get('compiled',[q]if'object'in q else[]):records[c['object']]=c;owners[c['object']]=q
 parent=docs['candidates/glm-rows16-model-r5707/CPU-build.json'];assert parent['ABI']==1 and parent['fresh_M_equal_MD']and sha(parent['library'])==parent['library_SHA256']==m['parent_DSO_SHA256']
 for path,h in parent['reused_objects_SHA256'].items():assert sha(path)==h,path
 for c in parent['compiled']:assert sha(c['object'])==c['object_SHA256']and sha(c['source'])==c['source_SHA256']
 for path,h in parent['dependency_SHA256'].items():assert sha(path)==h,path
 for path,h in parent['Runtime_and_extension_SHA256'].items():assert sha(path)==h,path
 for n,h in parent['SDK_libraries_SHA256'].items():assert sha(r/'install/hygon-glm-5f91ad7-r3672/lib'/n)==h,n
 assert sha(r/'candidates/glm-hybrid-decode-r5386/libnative.so')==parent['native_SHA256']==m['native_SHA256']
 reverse=json.loads((i/'r5730-GLM-ROWS32-actual-deps.json').read_text());assert reverse['receipt_SHA256']==m['receipt_SHA256']and reverse['current_Model_SHA256']==m['parent_DSO_SHA256']
 assert len(reverse['affected'])==6 and{pathlib.Path(x['source']).name for x in reverse['affected']}==set(m['target_TUs'])
 for n,alias in m['source_files'].items():shutil.copyfile(i/alias,base/'src'/n)
 replacements={};compiled=[];all_deps={};compiler=shutil.which('g++');original_commands={}
 for j,target in enumerate(reverse['affected']):
  old=records[target['object']];assert sha(old['object'])==target['object_SHA256']==old['object_SHA256']and sha(old['source'])==target['source_SHA256']==old['source_SHA256']
  assert sha(pathlib.Path(target['object']).with_suffix('.d'))==target['MD_SHA256']
  name=pathlib.Path(old['source']).name
  if name not in m['source_files']:shutil.copyfile(old['source'],base/'src'/name)
  a=next(list(a)for a in owners[target['object']]['commands']if isinstance(a,list)and'-c'in a and str(old['source'])in a and str(old['object'])in a)
  original_commands[name]=a;flags=[];k=1
  while k<len(a):
   if a[k]in['-c','-o','-MF','-MT','-MQ']:k+=2;continue
   if a[k]in['-MD','-MMD']:k+=1;continue
   flags.append(a[k]);k+=1
  flags=['-I'+str(base/'src'),*flags];assert '-D_GLIBCXX_USE_CXX11_ABI=1'in flags
  src=base/'src'/name;obj=base/f'rows32-{j}.o';dep=base/f'rows32-{j}.d'
  run([compiler,*flags,'-MD','-MF',str(dep),'-c',str(src),'-o',str(obj)],f'compile-{j}');ds=deps(run([compiler,*flags,'-M',str(src)],f'fresh-M-{j}'));assert ds==deps(dep.read_text())
  hits={p for p in ds if pathlib.Path(p).name=='glm_model_resident_policy_r5640.hpp'};assert hits=={str(base/'src/glm_model_resident_policy_r5640.hpp')},(name,hits)
  for p in ds:all_deps[p]=sha(p)
  compiled.append({'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'MD_file':str(dep),'MD_SHA256':sha(dep),'dependency_count':len(ds),'private_policy_header':sorted(hits),'source_changed':name in m['changed_source_TUs']})
  replacements[target['object']]=str(obj)
 args=next(list(a)for a in parent['commands']if'-shared'in a);assert all(args.count(a)==1 for a in replacements);original_objects=[a for a in args if a.endswith('.o')]
 assert len(original_objects)==26
 args=[replacements.get(a,a)for a in args];args[0]=compiler;lib=base/'libglm_rows32_model_r5734.so';args[args.index('-o')+1]=str(lib);run(args,'link')
 exports=run(['nm','-D','--defined-only',str(lib)],'exports');old=run(['nm','-D','--defined-only',parent['library']],'parent-exports');names=[a.split()[-1]for a in exports.splitlines()];oldnames={a.split()[-1]for a in old.splitlines()}
 api={n for n in oldnames if n.startswith('modern_glm_')or n.startswith('glm_grouped_')or n.startswith('glm_vec_')};assert api<=set(names);assert all(names.count(n)==1 for n in api)
 dyn=run(['readelf','-d',str(lib)],'dynamic');assert 'libglm_rows16_model_r5707'not in dyn and 'libglm_vector_model_r8542'not in dyn
 layout_src=base/'src/rows32_CPU_layout.cpp';layout_src.write_text('#include "glm_model_resident_policy_r5640.hpp"\n#include <iostream>\nint main(){std::cout<<sizeof(glm_model_resident_r5640::Slots)<<" "<<alignof(glm_model_resident_r5640::Slots)<<"\\n";}\n')
 run([compiler,'-std=c++17','-pthread','-I'+str(base/'src'),str(layout_src),'-o',str(base/'metadata_layout')],'metadata-layout-compile');layout=run([str(base/'metadata_layout')],'metadata-layout-run').strip()
 # This executable contains only standard-C++ metadata, never links/dlopens the GPU DSO.
 report.update(completed=True,ABI=1,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,source_manifest_SHA256=sha(mp),parent_DSO_SHA256=parent['library_SHA256'],parent_receipt_SHA256=m['receipt_SHA256'],compiled_TUs=6,compiled=compiled,dependency_SHA256=all_deps,fresh_M_equal_MD=True,header_ODR='All six actual MD policy consumers rebuilt against the same private Slots class definition; old fields/original method unchanged.',metadata_layout_DTK=layout,library=str(lib),library_SHA256=sha(lib),native_SHA256=parent['native_SHA256'],SDK_libraries_SHA256=parent['SDK_libraries_SHA256'],Runtime_and_extension_SHA256=parent['Runtime_and_extension_SHA256'],reused_objects_SHA256={a:sha(a)for a in args if a.endswith('.o')and a not in replacements.values()},ELF_dynamic=dyn,parent_API_export_count=len(api),original_compile_commands=original_commands,commands=commands,scope='Default ROWS64 and ROWS16 retained. Optional ROWS32 resident capacity32; M8 math unchanged, T OFF for ROWS32. CPU compile/link/metadata only, no GPU/weights/dlopen.',NOT_RUN=['CPUdlopen','GPUABI','ModelIDs','Modelperformance','ROWS32physicalallocation'])
except BaseException as exc:
 report.update(failure=repr(exc),commands=commands);(base/'CPU-failure.json').write_text(json.dumps(report,indent=2));raise
else:
 for p in[base/'CPU-build.json',r/'results/glm-rows32-CPU-r5734.json']:assert not p.exists();p.write_text(json.dumps(report,indent=2))
 print(json.dumps({k:report[k]for k in['completed','producer','compiled_TUs','elapsed_seconds','library_SHA256','parent_API_export_count','metadata_layout_DTK']}))
