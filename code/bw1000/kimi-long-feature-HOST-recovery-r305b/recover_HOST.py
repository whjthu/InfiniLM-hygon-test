"""Preserve CPU622 failure, reuse its compiled pair object, compile only remaining trace TU."""
from pathlib import Path
import os,json,hashlib,subprocess,time,sys,importlib.util,datetime
D=Path(__file__).resolve().parent;R=Path('/refactor');O=R/'incoming/Kimi-long-feature-capture-r305';A=R/'candidates/Kimi-long-feature-capture-r305';N=A/'native';B=R/'candidates/Kimi-long-feature-capture-r305b'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI_R305B_MANIFEST_SHA'];m=json.loads(mf.read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
 assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')and 'torch'not in sys.modules
 assert sha(O/'SOURCE_MANIFEST.json')=='80847e3526db29e4a2eb1fa33ebc4467c128be686931cabae933e18c5a08acb5';source=json.loads((O/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(O/n)==h for n,h in source['files_SHA256'].items())
 failed=A/'CPU-build.json';assert sha(failed)=='30411dcbd9c6e997a9c8b6110f88a14aa9e48f6cd31d7fc0e089a282d5c9d7a3';prior=json.loads(failed.read_bytes());assert prior['actual_CPU']=='910407.622'and not prior['passed']and prior['logs']['pair_probe-HOST']['rc']==prior['logs']['pair_probe-fresh-MD']['rc']==0
 assert sha(A/'pair_probe.o')=='4c72b06568d3748352febaa0c702e6a3aa4f3949dddca4a412e66f1e0c587717'
 for n,h in source['expected_patched_native_SHA256'].items():assert sha(N/n)==h
 index=json.loads((O/'baseline-native-index.json').read_bytes());assert sha(O/'baseline-native-index.json')=='b891594064ecc089f214d4aa2cf07922bef9f66002c705c53ad076ef084e3c5e'
 for n,h in index.items():
  rel=Path(n).relative_to('native')
  if str(rel)not in ['pair_probe.cpp','batch_live_taps_r9122.hpp']:assert sha(N/rel)==h
 assert not B.exists();B.mkdir();start=time.monotonic();q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_CPU':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'completed':False,'passed':False,'CPU_only':True,'GPU':0,'Model':0,'HTTP':0,'weights':0,'Native_linked':False,'Native_dlopen':False,'new_GPU_TUs':0,'new_HOST_TUs':1,'reused_HOST_TUs':1,'failed_CPU622_receipt_SHA256':sha(failed),'original_source_manifest_SHA256':sha(O/'SOURCE_MANIFEST.json'),'recovery_source_manifest_SHA256':sha(mf),'compiled_HOST_TUs':{},'logs':{}}
 def run(args,label,cap):
  left=220-(time.monotonic()-start);assert left>0;z=subprocess.run(args,capture_output=True,text=True,timeout=min(cap,left));p=B/(label+'.log');p.write_text(z.stdout+z.stderr);q['logs'][label]={'rc':z.returncode,'SHA256':sha(p)}
  if z.returncode:raise RuntimeError(label+' failed; see pinned log')
  return z.stdout
 try:
  spec=importlib.util.spec_from_file_location('fixed_HOST_functions',O/'build_HOST.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  ctxp=R/'incoming/Kimi-ctx2048-r123e/per-TU-original-context-r123e.json';assert sha(ctxp)=='07bb301b870f90e1ce9e8f771b2c73080937ee0d200ef753a7521fa81f922454';contexts=json.loads(ctxp.read_bytes())
  p=R/'candidates/Kimi-append512-r247/CPU-build.json';assert sha(p)=='21fc8ed69c6db1ec3a3c8925e61dc5f73112b5fea596a8caef977008da57046b';base247=json.loads(p.read_bytes())
  p=R/'candidates/Kimi-long-agent32K-r179/CPU-build.json';assert sha(p)=='37ac44052ffb3b0e43cf7e440178aad4096f09dcd97a8b12d6eca6fc65cd827d';base179=json.loads(p.read_bytes())
  for unit,name,base,oldN,oldObject in [('pair_probe','pair_probe.cpp',base247,'/refactor/incoming/Kimi-append512-r247/native','/refactor/candidates/Kimi-append512-r247/pair_probe.o'),('trace','kimi_live_trace_r9046.cpp',base179,'/refactor/incoming/Kimi-long-agent32K-r179/native','/refactor/candidates/Kimi-long-agent32K-r179/trace.o')]:
   flags=mod.semantic(contexts[unit]['argv']);assert Path(flags[0]).name=='g++'and '-O2'in flags and '-fPIC'in flags and '-pthread'in flags
   if unit=='pair_probe':obj=A/'pair_probe.o';dep=A/'pair_probe.d';fresh=A/'pair_probe.fresh.d';producer='910407.622'
   else:
    obj=B/'trace.o';dep=B/'trace.d';fresh=B/'trace.fresh.d';producer=q['actual_CPU'];prefix=['-I'+str(N),'-I'+str(N/'engine'),'-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine']
    run([flags[0],*prefix,*flags[1:],'-MD','-MF',str(dep),'-c',str(N/name),'-o',str(obj)],'trace-HOST',110);run([flags[0],*prefix,*flags[1:],'-M','-MF',str(fresh),'-MT',str(obj),str(N/name)],'trace-fresh-MD',25)
   md=mod.deps(dep);assert md==mod.deps(fresh);oldmd=base['all_new_MD'][unit];normalized={p.replace(str(N),oldN,1):p for p in md};newheaders={oldN+'/'+n for n in ['long_capture_geometry_r305.hpp','long_capture_api_r305.hpp']};assert set(normalized)-set(oldmd)==newheaders and not(set(oldmd)-set(normalized))
   changed={oldN+'/pair_probe.cpp'}if unit=='pair_probe'else{oldN+'/batch_live_taps_r9122.hpp'}
   for p,actual in normalized.items():
    if p not in newheaders|changed:assert sha(actual)==oldmd[p],p
   assert sha(oldObject)==base['all_linked_object_SHA256'][oldObject]
   # CAPI is externally defined symbols only; local compiler .cold labels are not exports.
   symbols=lambda text:{line.split()[-1]for line in text.splitlines()if len(line.split())>=3 and line.split()[-1].startswith('modern_')}
   oldsym=symbols(run(['nm','--extern-only','--defined-only',oldObject],unit+'-old-global-symbols',15));newsym=symbols(run(['nm','--extern-only','--defined-only',str(obj)],unit+'-new-global-symbols',15));expected={'modern_kimi_long_features_begin_r305','modern_kimi_long_features_capabilities_r305'}if unit=='pair_probe'else set();assert newsym-oldsym==expected and oldsym<=newsym
   if unit=='trace':text=run(['readelf','--wide','--syms',str(obj)],'trace-symbol-visibility',15);assert any('HIDDEN'in line and 'begin_long_validated_r305'in line for line in text.splitlines())
   q['compiled_HOST_TUs'][unit]={'producer_CPU':producer,'source_SHA256':sha(N/name),'object_path':str(obj),'object_SHA256':sha(obj),'original_effective_options_preserved':True,'original_context_SHA256':sha(ctxp),'fresh_MD_set_matches':True,'dependency_count':len(md),'new_dependencies':sorted(newheaders),'other_dependency_contents_equal':True,'old_public_CAPI_preserved':len(oldsym),'new_public_CAPI':sorted(expected),'actual_dependency_SHA256':{p:sha(p)for p in sorted(md)}}
  q.update(completed=True,passed=True,verifier_correction='Use nm --extern-only --defined-only; CPU622 failed only because new local .cold label was counted as public API. No Native source changed.',original_failed_receipt_preserved=True,GPU_GO=False,qualification='Two actual HOST object compiles across622/thisCPU plus source/dependency/global-symbol proof only; no Native link/load, GPU capture, draft integration, Model or speed qualification')
 except BaseException as e:q['error']=repr(e);raise
 finally:q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['actual_CPU','completed','passed','error','elapsed_s']}))
if __name__=='__main__':main()
