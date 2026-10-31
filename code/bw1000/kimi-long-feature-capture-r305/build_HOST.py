"""Compile two actual HOST TUs only. Never link/load Native or initialize a GPU."""
from pathlib import Path
import os,json,hashlib,subprocess,shlex,time,datetime,sys,signal
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/Kimi-long-feature-capture-r305'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def semantic(a):
 out=[a[0]];i=1
 while i<len(a):
  v=a[i]
  if v in ['-o','-MF','-MT','-MQ','-MJ']:i+=2;continue
  if v in ['-c','-MD','-MMD','-MP','-MG','-M','-MM','-shared']:i+=1;continue
  if v.startswith(('-MF','-MT','-MQ','-MJ','-Wl,','-L','-l'))or v.endswith(('.cpp','.cc','.cxx','.o','.so','.a')):i+=1;continue
  out.append(v);i+=1
 return out

def deps(p):return set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]))
def main():
 mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI_R305_MANIFEST_SHA'];m=json.loads(mf.read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
 assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD');assert 'torch'not in sys.modules
 assert not B.exists();B.mkdir();start=time.monotonic();q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_CPU':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'completed':False,'passed':False,'CPU_only':True,'GPU':0,'Model':0,'HTTP':0,'weights':0,'Native_linked':False,'Native_dlopen':False,'new_GPU_TUs':0,'compiled_HOST_TUs':{},'source_manifest_SHA256':sha(mf),'logs':{}}
 def run(args,label,cap):
  left=220-(time.monotonic()-start);assert left>0
  z=subprocess.run(args,capture_output=True,text=True,timeout=min(cap,left));p=B/(label+'.log');p.write_text(z.stdout+z.stderr);q['logs'][label]={'rc':z.returncode,'SHA256':sha(p)}
  if z.returncode:raise RuntimeError(label+' failed; see pinned log '+str(p))
  return z.stdout
 try:
  from apply_long_capture import patch
  N=B/'native';q['patch']=patch(R/'incoming/Kimi-append512-r247/native',N,D/'baseline-native-index.json');assert q['patch']['new_SHA256']==m['expected_patched_native_SHA256']
  ctxp=R/'incoming/Kimi-ctx2048-r123e/per-TU-original-context-r123e.json';assert sha(ctxp)=='07bb301b870f90e1ce9e8f771b2c73080937ee0d200ef753a7521fa81f922454';contexts=json.loads(ctxp.read_bytes())
  p=R/'candidates/Kimi-append512-r247/CPU-build.json';assert sha(p)=='21fc8ed69c6db1ec3a3c8925e61dc5f73112b5fea596a8caef977008da57046b';base247=json.loads(p.read_bytes());assert base247['completed']and base247['CPU_only']and not base247['Native_dlopen']
  p=R/'candidates/Kimi-long-agent32K-r179/CPU-build.json';assert sha(p)=='37ac44052ffb3b0e43cf7e440178aad4096f09dcd97a8b12d6eca6fc65cd827d';base179=json.loads(p.read_bytes());assert base179['completed']and base179['CPU_only']
  for unit,name,base,oldN,oldObject in [('pair_probe','pair_probe.cpp',base247,'/refactor/incoming/Kimi-append512-r247/native','/refactor/candidates/Kimi-append512-r247/pair_probe.o'),('trace','kimi_live_trace_r9046.cpp',base179,'/refactor/incoming/Kimi-long-agent32K-r179/native','/refactor/candidates/Kimi-long-agent32K-r179/trace.o')]:
   flags=semantic(contexts[unit]['argv']);assert Path(flags[0]).name=='g++'and '-O2'in flags and '-fPIC'in flags and '-pthread'in flags;assert not any(v in ['-xhip','--hip-link']or v.endswith('.cu')for v in flags)
   oldmd=base['all_new_MD'][unit];obj=B/(unit+'.o');prefix=['-I'+str(N),'-I'+str(N/'engine'),'-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine']
   argv=[flags[0],*prefix,*flags[1:],'-MD','-MF',str(B/(unit+'.d')),'-c',str(N/name),'-o',str(obj)];run(argv,unit+'-HOST',95)
   run([flags[0],*prefix,*flags[1:],'-M','-MF',str(B/(unit+'.fresh.d')),'-MT',str(obj),str(N/name)],unit+'-fresh-MD',25)
   md=deps(B/(unit+'.d'));assert md==deps(B/(unit+'.fresh.d'));normalized={p.replace(str(N),oldN,1):p for p in md}
   newheaders={oldN+'/'+n for n in ['long_capture_geometry_r305.hpp','long_capture_api_r305.hpp']};extra=set(normalized)-set(oldmd);missing=set(oldmd)-set(normalized)
   assert extra==newheaders and not missing,('unexpected dependency delta',sorted(extra),sorted(missing))
   changed={oldN+'/pair_probe.cpp'}if unit=='pair_probe'else{oldN+'/batch_live_taps_r9122.hpp'}
   for p,actual in normalized.items():
    if p not in newheaders|changed:assert sha(actual)==oldmd[p],('unintended dependency byte change',p)
   oldsha=base['all_linked_object_SHA256'][oldObject];assert sha(oldObject)==oldsha
   symbols=lambda s:{line.split()[-1]for line in s.splitlines()if len(line.split())>=3 and line.split()[-1].startswith('modern_')}
   oldsym=symbols(run(['nm','--defined-only',oldObject],unit+'-old-symbols',15));newsym=symbols(run(['nm','--defined-only',str(obj)],unit+'-new-symbols',15));expected_new={'modern_kimi_long_features_begin_r305','modern_kimi_long_features_capabilities_r305'}if unit=='pair_probe'else set();assert newsym-oldsym==expected_new and oldsym<=newsym
   if unit=='trace':
    symbols_text=run(['readelf','--wide','--syms',str(obj)],'trace-symbol-visibility',15);assert any('HIDDEN'in line and 'begin_long_validated_r305'in line for line in symbols_text.splitlines())
   q['compiled_HOST_TUs'][unit]={'source_SHA256':sha(N/name),'object_SHA256':sha(obj),'original_effective_compiler_options_preserved':True,'original_context_SHA256':sha(ctxp),'fresh_MD_set_matches':True,'dependency_count':len(md),'new_dependencies':sorted(extra),'other_dependency_contents_equal':True,'original_CAPI_exports_preserved':len(oldsym),'new_CAPI_exports':sorted(expected_new),'actual_dependency_SHA256':{p:sha(p)for p in sorted(md)}}
  q.update(completed=True,passed=True,new_HOST_TUs=2,GPU_GO=False,qualification='Actual runtime-header HOST compilation and object symbols only; no final Native link/load, device capture, state rollback, draft integration, Model or performance proof')
 except BaseException as e:q['error']=repr(e);raise
 finally:q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['actual_CPU','completed','passed','error','elapsed_s','new_HOST_TUs']}))
if __name__=='__main__':main()
