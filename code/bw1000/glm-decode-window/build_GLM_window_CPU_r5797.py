"""Actual DTK private4TU CPU compile/link only. No dlopen/GPU/weights."""
import pathlib,json,hashlib,subprocess,shlex,shutil,os,time
r=pathlib.Path('/refactor');i=r/'incoming';b=r/'candidates/glm-window-model-r5797';assert not b.exists();(b/'src').mkdir(parents=True);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((i/'GLM-window-CPU-manifest-r5797.json').read_text())
for n,h in m['sources_SHA256'].items():assert sha(i/n)==h,n
x=json.loads((i/m['input_receipt']).read_text());pp=r/'candidates/glm-layer77-model-r5775/CPU-build.json';assert sha(pp)==m['parent_receipt_SHA256'];p=json.loads(pp.read_text());assert p['completed']and sha(p['library'])==m['parent_DSO_SHA256']
cmds=[];receipt={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True};start=time.perf_counter()
def run(a,label):
 cmds.append(a);q=subprocess.run(a,text=True,capture_output=True);(b/(label+'.log')).write_text(q.stdout+q.stderr)
 if q.returncode:raise RuntimeError((label,q.returncode,q.stderr[-6000:]))
 return q.stdout
def deps(s):return set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]))
def flags(a):
 out=[];j=1
 while j<len(a):
  if a[j]in ['-c','-o','-MF','-MT','-MQ']:j+=2;continue
  if a[j]in ['-MD','-MMD']:j+=1;continue
  out.append(a[j]);j+=1
 return ['-I'+str(b/'src'),*out]
try:
 args=next(list(a)for a in p['commands']if '-shared'in a);objects=[v for v in args if v.endswith('.o')];original_sha={v:sha(v)for v in objects};assert len(objects)==27
 definition=run(['nm','-A','-C','--defined-only',*objects],'original-symbols');providers=[s for s in definition.splitlines()if ' T glm_layer_mark_r5670('in s];assert len(providers)==1,providers
 for n in ['glm_model_window_r5797.hpp','glm_model_window_r5797.cpp']:shutil.copyfile(i/n,b/'src'/n)
 compiled=[];replacements={};all_deps={}
 for j,t in enumerate(x['targets']+[{'record':{'source':'glm_model_window_r5797.cpp'},'compile_command':x['targets'][0]['compile_command']} ]):
  rec=t['record'];name=pathlib.Path(rec['source']).name;src=b/'src'/name
  if j<3:
   assert sha(rec['source'])==rec['source_SHA256']and sha(rec['object'])==rec['object_SHA256']and rec['object']in objects
   shutil.copyfile(i/'glm-window-r5797-src'/name,src)
  obj=b/f'window-{j}.o';dep=b/f'window-{j}.d';fl=flags(t['compile_command']);run([t['compile_command'][0],*fl,'-MD','-MF',str(dep),'-c',str(src),'-o',str(obj)],f'compile-{j}')
  ds=deps(dep.read_text());assert ds==deps(run([t['compile_command'][0],*fl,'-M',str(src)],f'fresh-M-{j}'));all_deps.update({v:sha(v)for v in ds});compiled.append({'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'MD_SHA256':sha(dep)})
  if j<3:replacements[rec['object']]=str(obj)
 args=[replacements.get(v,v)for v in args];args.insert(args.index('-o'),compiled[-1]['object']);lib=b/'libglm_window_model_r5797.so';args[args.index('-o')+1]=str(lib);run(args,'link');exports=run(['nm','-D','-C','--defined-only',str(lib)],'exports');assert sum(' T glm_layer_mark_r5670('in z for z in exports.splitlines())==1
 for sym in m['new_API']:assert sum(z.endswith(' '+sym)for z in exports.splitlines())==1,sym
 for v,h in original_sha.items():assert sha(v)==h
 receipt.update(completed=True,library=str(lib),library_SHA256=sha(lib),parent_DSO_SHA256=m['parent_DSO_SHA256'],parent_receipt_SHA256=m['parent_receipt_SHA256'],elapsed_seconds=time.perf_counter()-start,ABI=1,compiled=compiled,reused_objects_SHA256={v:h for v,h in original_sha.items()if v not in replacements},replaced_objects=replacements,sole_marker_provider_before=providers,fresh_M_equal_MD=True,dependency_SHA256=all_deps,commands=cmds,new_API=m['new_API'],scope='CPU compile/link/nm only; no GPU/dlopen/weights. Events512 fixed, stream queue-only observer; actual Model math untouched.')
except BaseException as e:receipt.update(error=repr(e),commands=cmds);(b/'CPU-failure.json').write_text(json.dumps(receipt,indent=2));raise
else:(b/'CPU-build.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:receipt[k]for k in ['completed','producer','elapsed_seconds','library','library_SHA256']}))
