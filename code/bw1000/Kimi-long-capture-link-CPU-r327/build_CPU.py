"""Reuse exact CPU622/623 objects in one isolated final Native link. No dlopen/GPU."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time,datetime
R=Path('/refactor');D=R/'incoming/Kimi-long-capture-link-CPU-r327';B=R/'candidates/Kimi-long-capture-link-CPU-r327';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')and'torch'not in sys.modules
 mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI_LONG_LINK_MANIFEST_SHA'];m=json.loads(mf.read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
 assert not B.exists();B.mkdir();start=time.monotonic();q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_CPU':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'source_manifest_SHA256':sha(mf),'completed':False,'passed':False,'CPU_only':True,'GPU':0,'weights':0,'HTTP':0,'Native_dlopen':False,'new_HOST_TUs':0,'new_GPU_TUs':0,'Native_linked':False,'logs':{}}
 def run(args,label,cap):
  left=205-(time.monotonic()-start);assert left>0;z=subprocess.run(args,capture_output=True,text=True,timeout=min(cap,left));p=B/(label+'.log');p.write_text(z.stdout+z.stderr);q['logs'][label]={'rc':z.returncode,'SHA256':sha(p)}
  if z.returncode:raise RuntimeError(label+' failed; retained log')
  return z.stdout
 try:
  receipts={}
  for name,rel,h in [('base','candidates/Kimi-append512-r247/CPU-build.json','21fc8ed69c6db1ec3a3c8925e61dc5f73112b5fea596a8caef977008da57046b'),('host','candidates/Kimi-long-feature-capture-r305b/CPU-build.json','5d32794a8cf4e95221f48422d1d34e3d859aaab71b303b4ce20dff1ea8e665ef'),('recipe','candidates/Kimi-W2-pair-r9280/CPU-build.json','e86d3428b6134873f5525cbfd0409168593ec1d17c2fa3b48bfcebfa46ebf86e')]:
   p=R/rel;assert sha(p)==h;receipts[name]=json.loads(p.read_bytes())
  base,host,recipe=[receipts[n]for n in ['base','host','recipe']];assert base['completed']and base['CPU_only']and not base['Native_dlopen'];assert base['native_SHA256']=='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'and sha(base['native'])==base['native_SHA256']
  assert host['completed']and host['passed']and host['actual_CPU']=='910407.623'and not host['Native_linked']and not host['Native_dlopen']
  for path,h in base['all_linked_object_SHA256'].items():assert sha(path)==h
  for entry in base['actual_dependency_library_SHA256'].values():assert sha(entry['path'])==entry['SHA256']
  replacements={};deps={}
  for unit,old in [('pair_probe','/refactor/candidates/Kimi-append512-r247/pair_probe.o'),('trace','/refactor/candidates/Kimi-long-agent32K-r179/trace.o')]:
   v=host['compiled_HOST_TUs'][unit];assert v['fresh_MD_set_matches']and v['other_dependency_contents_equal']and sha(v['object_path'])==v['object_SHA256'];replacements[old]=v['object_path']
   for path,h in v['actual_dependency_SHA256'].items():
    assert path not in deps or deps[path]==h;deps[path]=h
  for path,h in deps.items():assert sha(path)==h
  q['fresh_dependency_paths_rechecked']=len(deps)
  link=next(c[:]for c in recipe['commands']if '-shared'in c and '-o'in c and c[c.index('-o')+1]==recipe['native'])
  mapping={'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/pair_probe.o':'/refactor/candidates/Kimi-append512-r247/pair_probe.o','/refactor/candidates/Kimi-speculative-batch-Model-r9122c/worker.o':'/refactor/candidates/Kimi-long-agent32K-r179/worker.o','/refactor/candidates/Kimi-speculative-batch-Model-r9122c/delta.o':'/refactor/candidates/Kimi-long-agent32K-r179/delta.o','/refactor/candidates/Kimi-speculative-batch-Model-r9122c/trace.o':'/refactor/candidates/Kimi-long-agent32K-r179/trace.o','/refactor/candidates/kimi-compact-cache-r4974/objects/infer_engine.o':'/refactor/candidates/Kimi-long-agent32K-r179/infer_engine.o','/refactor/candidates/kimi-compact-cache-r4974/objects/kimi_k3_allocate_cache.o':'/refactor/candidates/Kimi-long-agent32K-r179/allocate.o','/refactor/candidates/Kimi-I192-batch-Model-r9129/moe.o':'/refactor/candidates/Kimi-long-scale-exact-r201/moe.o'}
  for old,new in mapping.items():assert link.count(old)==1;link[link.index(old)]=new
  assert {x for x in link if x.endswith('.o')}==set(base['all_linked_object_SHA256'])
  for old,new in replacements.items():assert link.count(old)==1;link[link.index(old)]=new
  native=B/'libKimi_long_capture_r327.so';link[link.index('-o')+1]=str(native)
  for i,v in enumerate(link):
   if v.startswith('-Wl,-Map'):link[i]='-Wl,-Map,'+str(B/'link.map')
  run(link,'isolated-Native-link',70);q['Native_linked']=True
  symbols=lambda text:{line.split()[-1]for line in text.splitlines()if len(line.split())>=3 and line.split()[-1].startswith('modern_')}
  oldsymbols=symbols(run(['nm','-D','--defined-only',base['native']],'baseline-exports',15));newsymbols=symbols(run(['nm','-D','--defined-only',str(native)],'candidate-exports',15))
  added={'modern_kimi_long_features_begin_r305','modern_kimi_long_features_capabilities_r305'};assert oldsymbols<=newsymbols and newsymbols-oldsymbols==added
  dynamic=run(['readelf','--wide','-d',str(native)],'candidate-dynamic',15);old_dynamic=run(['readelf','--wide','-d',base['native']],'baseline-dynamic',15)
  needed=lambda text:sorted(x.split('[')[1].split(']')[0]for x in text.splitlines()if'(NEEDED)'in x);assert needed(dynamic)==needed(old_dynamic)
  resolved=run(['ldd',str(native)],'candidate-ldd',20);assert 'not found'not in resolved
  q.update(completed=True,passed=True,native=str(native),native_SHA256=sha(native),baseline_native_SHA256=base['native_SHA256'],original_link_recipe_receipt_SHA256=m['recipe_receipt_SHA256'],original_effective_link_options_retained=True,baseline_linked_objects=base['all_linked_object_SHA256'],replaced_HOST_objects={old:{'path':new,'SHA256':sha(new)}for old,new in replacements.items()},actual_dependency_library_SHA256=base['actual_dependency_library_SHA256'],retained_public_CAPI_count=len(oldsymbols),new_public_CAPI=sorted(added),DT_NEEDED=needed(dynamic),dynamic_dependency_names_unchanged=True,qualification='Isolated final Native link/source/dependencies/exports only. No Native dlopen, actual Context/devicecapture, Pythonadapter, Model or speed qualification.',default_OFF=True,current_AR_affected=False)
 except BaseException as e:q['error']=repr(e)
 finally:q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','error','elapsed_s','native_SHA256']}),flush=True)
 assert q['passed']and'torch'not in sys.modules
if __name__=='__main__':main()
