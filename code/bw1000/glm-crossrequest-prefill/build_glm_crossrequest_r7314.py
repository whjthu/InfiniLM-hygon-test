"""One private CPU-only adapter TU; no dlopen, GPU or checkpoint I/O."""
import pathlib,json,hashlib,subprocess,shutil,shlex,time,os
r=pathlib.Path('/refactor');base=r/'candidates/glm-crossrequest-attention-r7314';assert not base.exists();(base/'src').mkdir(parents=True)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
manifest=json.loads((r/'incoming/GLM-crossrequest-CPU-manifest-r7314.json').read_text())
source=r/'incoming/glm_crossrequest_attention_r7314.cpp';assert sha(source)==manifest['source_SHA256'];src=base/'src'/source.name;shutil.copyfile(source,src)
report={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,'commands':[]};start=time.perf_counter()
def run(a,label):
 report['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True);(base/(label+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError((label,p.returncode,p.stderr[-6000:]))
 return p.stdout
try:
 parent=r/'candidates/glm-rows16-model-r5707/libglm_rows16_model_r5707.so';assert sha(parent)==manifest['parent_DSO_SHA256']
 receipt=r/'candidates/glm-transient-layer-r5681/CPU-build.json';assert sha(receipt)==manifest['layer_receipt_SHA256'];q=json.loads(receipt.read_text());a=next(a for a in q['commands']if'-c'in a and any('dense_modes_r48.cpp'in s for s in a))
 flags=[];i=1
 while i<len(a):
  if a[i]in['-c','-o','-MF','-MT','-MQ']:i+=2;continue
  if a[i]in['-MD','-MMD']or a[i]=='-I/refactor/incoming':i+=1;continue
  flags.append(a[i]);i+=1
 obj=base/'adapter.o';md=base/'adapter.d';compiler=shutil.which('g++')
 run([compiler,*flags,'-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],'compile')
 parse=lambda s:set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]))
 deps=parse(md.read_text());assert deps==parse(run([compiler,*flags,'-M',str(src)],'fresh-M'));assert not any('/incoming/'in d for d in deps)
 expected=r/'candidates/glm-deferred-attention-model-5f91ad7-r5096/src/glm_moe_dsa_sparse_attention.hpp';assert str(expected)in deps
 p=json.loads((r/'candidates/glm-rows16-model-r5707/CPU-build.json').read_text());oldlink=next(a for a in p['commands']if'-shared'in a)
 # Reuse exact actual link libraries/searchpaths; substitute only objects with parent DSO+adapter.
 link=[compiler,str(obj),str(parent),'-std=c++17','-O2','-shared','-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1'];i=1
 while i<len(oldlink):
  v=oldlink[i]
  if v=='-o':i+=2;continue
  if v.startswith(('-L','-l','-Wl,'))and v!='-Wl,-Bsymbolic':link.append(v)
  i+=1
 lib=base/'libglm_crossrequest_r7314.so';link+=['-Wl,-rpath,'+str(parent.parent),'-o',str(lib)];run(link,'link')
 exports=run(['nm','-D','--defined-only',str(lib)],'exports');undefined=run(['nm','-D','--undefined-only',str(lib)],'undefined');dynamic=run(['readelf','-d',str(lib)],'dynamic')
 api=['glm_crossrequest_'+s+'_r7314'for s in['create','forward','destroy','error']];assert all(sum(line.endswith(' '+n)for line in exports.splitlines())==1 for n in api)
 assert 'libglm_rows16_model_r5707.so'in dynamic
 report.update(completed=True,elapsed_seconds=time.perf_counter()-start,source=str(src),source_SHA256=sha(src),object=str(obj),object_SHA256=sha(obj),MD_SHA256=sha(md),fresh_M_equal_MD=True,dependency_SHA256={d:sha(d)for d in deps},parent_DSO_SHA256=sha(parent),library=str(lib),library_SHA256=sha(lib),Runtime_and_extension_SHA256=p['Runtime_and_extension_SHA256'],SDK_libraries_SHA256=p['SDK_libraries_SHA256'],ABI=1,exports=exports,undefined=undefined,dynamic=dynamic,NOT_RUN=['CPUdlopen','GPU','Model','performance'])
 for path,h in report['Runtime_and_extension_SHA256'].items():assert sha(path)==h
 for name,h in report['SDK_libraries_SHA256'].items():assert sha(r/'install/hygon-glm-5f91ad7-r3672/lib'/name)==h
except BaseException as e:
 report['error']=repr(e);(base/'CPU-failure.json').write_text(json.dumps(report,indent=2));raise
else:
 (base/'CPU-build.json').write_text(json.dumps(report,indent=2));(r/'results/glm-crossrequest-CPU-r7314.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k]for k in['completed','producer','elapsed_seconds','library_SHA256']}))
