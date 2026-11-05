"""One changedHOST TU+DSO link only. No SDK/Runtime/Common constructor executes."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback,shlex,re,runpy
D=Path(__file__).resolve().parent;P=D.parent;O=Path('/refactor/candidates/GLM-global-TP16-phase-r470-CPU-v3');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deps(s):return set(shlex.split(s.replace(chr(92)+chr(10),' ').partition(':')[2]))
def main():
 start=time.monotonic();assert not O.exists();O.mkdir();q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=False,Runtime=False,Native_dlopen=False,Common_dlopen=False,Model=False,weights=False,new_HOST=0,new_GPU=0,commands=[])
 def save():(O/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(a,name):
  left=265-(time.monotonic()-start);assert left>5;q['commands'].append(dict(name=name,argv=a));save();z=subprocess.run(a,capture_output=True,text=True,timeout=left-3);(O/(name+'.stdout')).write_text(z.stdout);(O/(name+'.stderr')).write_text(z.stderr);assert z.returncode==0,(name,z.returncode,z.stderr[-2500:]);return z.stdout
 try:
  assert os.environ['SLURM_JOB_ID']=='910407'and all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);assert H(D/'CPU_RESOURCE_CLAIM.json')==os.environ['GLM470PHASE_CLAIM_SHA'];claim=json.loads((D/'CPU_RESOURCE_CLAIM.json').read_bytes());assert claim['CPU_GO']and not claim['GPU_GO']and H(D/'SOURCE_MANIFEST.json')==claim['source_manifest_SHA256'];assert all(H(D/n)==h for n,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items());reuse=json.loads((D/'REUSE.json').read_bytes());assert H(P/'BUILD_PIN.json')==reuse['parent_build_pin_SHA256']and H(P/'TOOLCHAIN.json')==reuse['parent_toolchain_SHA256']and H(P/'component.cpp')==reuse['parent_component_SHA256']and H(D/'geometry.hpp')==H(P/'geometry.hpp')==reuse['parent_geometry_SHA256'];assert H(reuse['source1061_receipt'])==reuse['source1061_receipt_SHA256']and H(reuse['source1061_library'])==reuse['source1061_library_SHA256'];prior=json.loads(Path(reuse['source1061_receipt']).read_bytes());assert prior['passed']and prior['actual_CPU']=='910407.1061'
  for path,h in reuse['objects'].items():assert H(path)==h
  pin=json.loads((P/'BUILD_PIN.json').read_bytes());libs={**pin['libraries'],reuse['Common']:reuse['Common_SHA256']}
  for p,h in libs.items():assert H(p)==h,p
  for p,v in json.loads((P/'TOOLCHAIN.json').read_bytes())['tools'].items():assert str(Path(p).resolve())==v['canonical']and H(p)==v['sha256']
  flags=pin['compile_flags'];source=D/'component.cpp';obj=O/'component.o';md=O/'component.d';text=run(['/usr/bin/g++',*flags,'-I'+str(D),'-M',str(source)],'phase-M');before={p:H(p)for p in deps(text)};q['fresh_dependency_SHA256']=before;save()
  for p,h in before.items():assert(Path(p).parent==D and H(p)==h)or pin['dependency_SHA256'].get(p)==h,('Unpinned dependency',p,h)
  run(['/usr/bin/g++',*flags,'-I'+str(D),'-MD','-MF',str(md),'-c',str(source),'-o',str(obj)],'phase-compile');assert deps(md.read_text())==set(before)and all(H(p)==h for p,h in before.items());q['new_HOST']=1
  lib=O/'libGLM_global_TP16_phase_r470.so';link=list(next(v['argv']for v in prior['commands']if v['name']=='component-link'));oldobj=next(p for p in reuse['objects']if p.endswith('/component.o'));assert oldobj in link;link[link.index(oldobj)]=str(obj);link[link.index('-o')+1]=str(lib);run(link,'phase-link');elf=run(['/usr/bin/readelf','-d',str(lib)],'phase-ELF');oldelf=run(['/usr/bin/readelf','-d',reuse['source1061_library']],'parent-ELF');needed=lambda s:re.findall(r'\(NEEDED\).*?\[([^\]]+)\]',s);assert needed(elf)==needed(oldelf)and not re.search(r'\(SONAME\)',elf);nm=run(['/usr/bin/nm','-D','--defined-only',str(lib)],'phase-CABI');symbols={v.split()[-1]for v in nm.splitlines()};assert {'glm_global_tp16_'+n+'_r470'for n in ['error','id_size','id','create','geometry','probe','close','trace_path']}<=symbols
  out=run(['/usr/bin/ldd','-r',str(lib)],'phase-loader-metadata');err=(O/'phase-loader-metadata.stderr').read_text();ldd=out+'\n'+err;assert not re.search(r'not found|undefined symbol',ldd,re.I);entries=[]
  for line in ldd.splitlines():
   m=re.match(r'\s*(\S+)\s+=>\s+(/\S+)\s+\(',line)
   if m:entries.append((m[1],m[2]))
  for p,h in libs.items():match=[v for name,v in entries if name==Path(p).name];assert len(match)==1 and H(match[0])==h,(p,match)
  typed=runpy.run_path(str(D/'binding_trace.py'))['check_new_ffi']();assert typed['passed'];q.update(passed=True,library=str(lib),library_SHA256=H(lib),objects_SHA256={str(obj):H(obj),str(md):H(md)},source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),claim_SHA256=H(D/'CPU_RESOURCE_CLAIM.json'),verified_libraries=libs,fresh_M_equals_MD=True,retained1061_object_metadata=reuse['objects'],new_CABI=1,typed_trace=typed,old_geometry_suite_repeated=False,old_typed_suite_repeated=False,component_executed=False,ldd_stdout_SHA256=H(O/'phase-loader-metadata.stdout'),ldd_stderr_SHA256=H(O/'phase-loader-metadata.stderr'),loader_resolution=entries,NEEDED=needed(elf))
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:q.update(completed=True,seconds=time.monotonic()-start);save()
 print(json.dumps({k:q.get(k)for k in ['passed','actual_CPU','seconds','error','library_SHA256']}))
if __name__=='__main__':main()
