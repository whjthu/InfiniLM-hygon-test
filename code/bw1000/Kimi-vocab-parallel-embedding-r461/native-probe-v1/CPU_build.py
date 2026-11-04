"""Only changed pair+worker HOST, fixed sidecar CPU leaf and private relink.

No Native/SDK dlopen or execution; CPU981 textModel/GPU objects are reused.
"""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;P=D.parent;B=Path('/refactor/candidates/Kimi-vocab-parallel-embedding-r461-probe-CPU-v1')
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def semantic(a):
 out=[a[0]];i=1
 while i<len(a):
  z=a[i]
  if z in ['-o','-MF','-MT','-MQ','-MJ']:i+=2;continue
  if z in ['-c','-MD','-MMD','-MP','-MG','-M','-MM','-shared']:i+=1;continue
  if z.startswith(('-MF','-MT','-MQ','-MJ','-Wl,','-L','-l'))or z.endswith(('.cpp','.cc','.cxx','.o','.so','.a')):i+=1;continue
  out.append(z);i+=1
 return out
def deps(p):return set(shlex.split(Path(p).read_text().replace('\\\n',' ').split(':',1)[1]))
def main():
 assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 assert not os.environ.get('LD_PRELOAD')and 'torch'not in sys.modules
 assert not B.exists();B.mkdir();start=time.monotonic()
 q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Model=0,Native_dlopen=False,SDK_execution=False,
  new_HOST_TUs=2,new_GPU_TUs=0,pure_sidecar_contract_TUs=1,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
  source_manifest_SHA256=H(D/'CPU_SOURCE_MANIFEST.json'),commands=[],logs={})
 def save():(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(args,label,cap=90):
  left=270-(time.monotonic()-start);assert left>0
  q['phase']=label;q['commands'].append(args);save()
  p=subprocess.run(args,capture_output=True,text=True,timeout=min(left,cap));log=B/(label+'.log');log.write_text(p.stdout+p.stderr)
  q['logs'][label]=dict(rc=p.returncode,SHA256=H(log));save()
  if p.returncode:raise RuntimeError(label+' failed; original '+str(log))
  return p.stdout
 try:
  m=json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())
  for name,h in m['files_SHA256'].items():assert H(D/name)==h,name
  oldpath=Path('/refactor/candidates/Kimi-vocab-parallel-embedding-r461-CPU-v2/CPU-result.json')
  assert H(oldpath)=='052818841ad38d97114e8a5e16634f67f67b743e43070fcccfbfeab20233327e'
  base=json.loads(oldpath.read_bytes());assert base['passed']and base['actual_CPU']=='910407.981'
  parentpath=Path('/refactor/candidates/Kimi-AR-persistent-storage-optin-r456-CPU-v2/CPU-result.json')
  assert H(parentpath)=='cfe2237631aab314d8f410aceef029de486361edbc76c80ab421bb7655bb0185'
  parent=json.loads(parentpath.read_bytes())
  for p,h in base['all_linked_object_SHA256'].items():assert H(p)==h,p
  assert H(base['native'])==base['native_SHA256']=='fa1f052fa08884a397cf2236daad775f8e1390babf7127a5941edae5c96571bb'
  for v in base['actual_dependency_library_SHA256'].values():assert H(v['path'])==v['SHA256']
  from check_source import check
  q['source_contract']=check();save()
  contract=B/'CPU-sidecar-contract'
  run(['g++','-std=c++17','-O2','-pthread','-Wall','-Wextra','-Werror','-I'+str(D),str(D/'CPU_protocol_contract.cpp'),'-o',str(contract)],'sidecar-contract-compile',20)
  dynamic=run(['readelf','-d',str(contract)],'sidecar-contract-ELF',5)
  assert not any(s in dynamic for s in ['libinfinilm','libinfiniops','libinfinirt','libcuda','libcudart'])
  q['actual_sidecar_contract']=json.loads(run([str(contract)],'actual-fixed-sidecar-contract',15));assert q['actual_sidecar_contract']['passed']
  objects={};details={};old_sources={}
  for unit,receipt,original in [('pair_probe',base,'pair_probe.cpp'),('worker',parent,'worker.cpp')]:
   argv=next(c for c in receipt['commands']if '-c'in c and Path(c[c.index('-c')+1]).name==original)
   old_sources[unit]=argv[argv.index('-c')+1]
   prefix=['g++','-I'+str(D),'-I'+str(P),*semantic(argv)[1:]]
   source=D/'native'/original;before=B/(unit+'.before.d');actual=B/(unit+'.d');obj=B/(unit+'.o')
   run([*prefix,'-M','-MF',str(before),'-MT',str(obj),str(source)],unit+'-before-M',25)
   snapshot={p:H(p)for p in sorted(deps(before))}
   run([*prefix,'-MD','-MF',str(actual),'-c',str(source),'-o',str(obj)],unit+'-compile',90)
   assert deps(actual)==set(snapshot)and all(H(p)==h for p,h in snapshot.items())
   assert str(D/'probe_protocol.hpp')in snapshot
   if unit=='worker':assert str(D/'probe_owned.hpp')in snapshot and str(P/'vocab_parallel_embedding_r461.hpp')in snapshot
   oldmd=base['compiled']['pair_probe']['all_new_MD']if unit=='pair_probe'else parent['HOST']['worker']['all_new_MD']
   for p,h in oldmd.items():
    if p==old_sources[unit]:continue
    assert snapshot.get(p)==h,(unit,'old MD changed/omitted',p)
   objects[unit]=str(obj);details[unit]=dict(source=str(source),source_SHA256=H(source),object_SHA256=H(obj),all_new_MD=snapshot,MD_before_after_identical=True)
  q['compiled']=details
  oldpair=str(oldpath.parent/'pair_probe.o');oldworker=str(parentpath.parent/'worker.o')
  replaced={oldpair:objects['pair_probe'],oldworker:objects['worker']}
  link=next(c[:]for c in base['commands']if '-shared'in c);assert set(replaced)<=set(link)
  link=[replaced.get(v,v)for v in link];native=B/'libKimi_vocab_parallel_embedding_probe_r461.so';link[link.index('-o')+1]=str(native)
  run(link,'private-Native-link',40)
  linked=[p for p in link if p.endswith('.o')];assert len(linked)==34
  reused={p:h for p,h in base['all_linked_object_SHA256'].items()if p not in replaced};assert len(reused)==32
  assert all(p in linked and H(p)==h for p,h in reused.items())
  def strong(path,label):
   rows=run(['nm','-D','--defined-only',str(path)],label,15)
   return {w[-1]for line in rows.splitlines()if len(w:=line.split())>=3 and w[-2]in ['T','D','B','R']}
  old=strong(base['native'],'CPU981-strong-symbols');new=strong(native,'new-strong-symbols');assert old<=new
  required={'modern_kimi_embedding_probe_load_r461','modern_kimi_embedding_probe_r461'};assert required<=new
  dynamic=run(['readelf','--wide','-d',str(native)],'Native-dynamic',10);assert 'SYMBOLIC'in dynamic
  ldd=run(['ldd','-r',str(native)],'Native-ldd-relocations-only',20);assert 'not found'not in ldd and 'undefined symbol'not in ldd
  q.update(completed=True,passed=True,native=str(native),native_SHA256=H(native),all_linked_object_SHA256={p:H(p)for p in linked},unchanged_objects_SHA256=reused,
   old_strong_exports_retained=len(old),old_CAPI_retained=len({n for n in old if n.startswith('modern_')}),new_CAPI=sorted(required),actual_dependency_library_SHA256=base['actual_dependency_library_SHA256'])
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:
  q.update(elapsed_s=time.monotonic()-start,Torch_imported='torch'in sys.modules);save()
  print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','error','native_SHA256','elapsed_s']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
