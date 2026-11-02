"""One host dispatch TU, unchanged038d link objects/GPU kernels/SDK; no dlopen."""
from pathlib import Path
import os,json,hashlib,subprocess,time,traceback,re
R=Path('/refactor');D=R/'incoming/GLM-smallM-FFN-feasibility-r367';B=R/'candidates/GLM-smallM-FFN-r367';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q=dict(completed=False,CPU_only=True,GPU_initialized=False,dlopen=False,weights_opened=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[])
def run(args):
 q['commands'].append(args);p=subprocess.run(args,text=True,capture_output=True,timeout=180);(B/('command%d.log'%len(q['commands']))).write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-3000:];return p.stdout
try:
 mf=D/'CPU_SOURCE_MANIFEST.json';m=json.loads(mf.read_bytes());assert sha(mf)==os.environ['GLM367_CPU_SOURCE_SHA256']
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 q['source_manifest_SHA256']=sha(mf)
 p=R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json';assert sha(p)=='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529';parent=json.loads(p.read_bytes());assert parent['completed']and parent['library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'and sha(parent['library'])==parent['library_SHA256']
 old=R/'candidates/GLM-down-row2-Model-r9284/dispatch.o';assert sha(old)=='a421246080486aa95c38964575a0e44b16d85d4e6022b750f06abe9fabc8c06f'
 q['parent_CPU_receipt_SHA256']=sha(p);q['parent_native_SHA256']=parent['library_SHA256']
 original=json.loads((R/'candidates/GLM-down-row2-Model-r9284/CPU-build.json').read_bytes())
 command=next(list(a)for a in original['commands']if isinstance(a,list)and'-c'in a and any(x.endswith('/dispatch-row2-Model-r9284.cpp')for x in a))
 source=command[command.index('-c')+1];assert sha(source)==sha(D/'baseline_dispatch_r9284.cpp')=='2bedb52f033e73cdc68addaa6c3ec893802c8e3e36980a840b397083bddd3ebf'
 header=D/'glm_vector_dispatch_r8540.hpp';assert sha(header)=='2e5b8d6bdd117fd21ab1920f594570d804ce3338a83605ae63d293dc8db2d6c7'
 # Actual selector, stub GPU provider calls. No Runtime constructor or Native load.
 test=B/'check_dispatch_CPU';run(['/usr/bin/g++','-std=c++17','-O2','-I'+str(D),str(D/'dispatch_M1_r367.cpp'),str(D/'check_dispatch_CPU.cpp'),'-o',str(test)])
 contract=json.loads(run([str(test)]));assert contract['passed']and contract['actual_dispatch_checks']==832;q['CPU_contract']=contract
 obj=B/'dispatch_M1.o';command[command.index('-c')+1]=str(D/'dispatch_M1_r367.cpp');command[command.index('-o')+1]=str(obj)
 if '-MF'in command:command[command.index('-MF')+1]=str(B/'dispatch_M1.d')
 else:command+=['-MMD','-MF',str(B/'dispatch_M1.d')]
 command.insert(1,'-I'+str(D));run(command);q['HOST_TUs_compiled']=1
 md=B/'dispatch_M1.d';assert md.exists();deps=[x for x in md.read_text().replace('\\\n',' ').split(':',1)[1].split()]
 q['dependency_SHA256']={p:sha(p)for p in deps};q['MD_SHA256']=sha(md)
 link=next(list(a)for a in parent['commands']if isinstance(a,list)and'-shared'in a)
 reused={x:sha(x)for x in link if x.endswith('.o')};assert str(old)in reused and reused[str(old)]=='a421246080486aa95c38964575a0e44b16d85d4e6022b750f06abe9fabc8c06f'
 for p,h in parent['reused_objects_SHA256'].items():assert reused[p]==h,p
 lib=B/'libGLM_M1_gate_r367.so';link=[str(obj)if x==str(old)else x for x in link];link[link.index('-o')+1]=str(lib);run(link)
 oldnm=run(['nm','-D','--defined-only',parent['library']]);nm=run(['nm','-D','--defined-only',str(lib)])
 symbols=lambda s:{line.split()[-1]for line in s.splitlines()if line.strip()}
 assert symbols(oldnm)<=symbols(nm)and all(n in symbols(nm)for n in ['modern_glm_M1_gate_policy_r367','modern_glm_M1_gate_counts_r367','glm_layer_forward_r5670','glm_layer_create_r5670'])
 def unresolved(lib):
  q['commands'].append(['ldd','-r',lib]);p=subprocess.run(['ldd','-r',lib],text=True,capture_output=True,timeout=45);assert p.returncode==0 and'undefined symbol'not in p.stdout+p.stderr and'not found'not in p.stdout+p.stderr;return p.stdout
 parent_ldd=unresolved(parent['library']);candidate_ldd=unresolved(str(lib))
 dependencies={}
 for text in [parent_ldd,candidate_ldd]:
  for p in re.findall(r'=>\s+(/[^\s]+)',text):dependencies[p]=sha(p)
 assert any(h=='0bd37990340f39bbcaf185abbc990ebeb9f56ea1b9992327eda96d627446a863'for h in dependencies.values())
 for p,h in reused.items():assert sha(p)==h,p
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),source_SHA256=sha(D/'dispatch_M1_r367.cpp'),object_SHA256=sha(obj),replaced_object=dict(path=str(old),SHA256=reused.pop(str(old))),reused_objects_SHA256=reused,actual_dynamic_dependency_SHA256=dependencies,all_parent_exports_retained=True,Frame_ABI_unchanged=True,default_OFF=True,GPU_primitive_recompiled=False,gate_provider_changed_for_M1=True,quantization_or_arithmetic_formula_changed=False,CPU_scope='Actual dispatch state/branch/counter/error contract plus current038d host relink; GPU numerics/full-component/Model not qualified')
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
q['elapsed_seconds']=time.monotonic()-start
with(B/'CPU-build.json').open('x')as f:json.dump(q,f,indent=2)
print(json.dumps({k:q.get(k)for k in ['completed','producer','CPU_contract','error','library','library_SHA256','elapsed_seconds']}));raise SystemExit(0 if q['completed']else 1)
