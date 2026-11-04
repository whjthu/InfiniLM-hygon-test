"""Five necessary HOST includers, 28 exact CPU961 objects reused; no SDK run."""
from pathlib import Path
import hashlib,importlib.util,json,os,shlex,subprocess,sys,time
D=Path(__file__).resolve().parent;N=D/'native';P=Path('/refactor/incoming/Kimi-long-geometry-optin-r453');B=Path('/refactor/candidates/Kimi-AR-persistent-storage-optin-r456-CPU-v1')
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deps(p):return set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]))
def canonical(p):return os.path.normpath(p)
def semantic(a):
 out=[a[0]];i=1
 while i<len(a):
  z=a[i]
  if z in ['-o','-MF','-MT','-MQ','-MJ']:i+=2;continue
  if z in ['-c','-MD','-MMD','-MP','-MG','-M','-MM','-shared']:i+=1;continue
  if z.startswith(('-MF','-MT','-MQ','-MJ','-Wl,','-L','-l'))or z.endswith(('.cpp','.cc','.cxx','.o','.so','.a')):i+=1;continue
  out.append(z);i+=1
 return out
def main():
 assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
 assert 'torch'not in sys.modules and 'transformers'not in sys.modules
 m=json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())
 for p,h in m['files_SHA256'].items():assert H(D/p)==h,p
 assert not B.exists();B.mkdir();start=time.monotonic()
 q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Model=0,GPU_GO=False,Model_GO=False,Native_dlopen=False,Runtime_created=False,new_HOST_TUs=5,new_GPU_TUs=0,pure_contract_TUs=1,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=H(D/'CPU_SOURCE_MANIFEST.json'),commands=[],logs={})
 def save():(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(a,label,cap=90):
  left=270-(time.monotonic()-start);assert left>0;q['phase']=label;q['commands'].append(a);save()
  p=subprocess.run(a,capture_output=True,text=True,timeout=min(left,cap));log=B/(label+'.log');log.write_text(p.stdout+p.stderr);q['logs'][label]=dict(rc=p.returncode,SHA256=H(log))
  if p.returncode:raise RuntimeError(label+' failed; '+str(log))
  return p.stdout
 def receipt(p,h):assert H(p)==h,p;v=json.loads(Path(p).read_bytes());assert v['completed']and v['CPU_only'];return v
 try:
  base=receipt('/refactor/candidates/Kimi-long-geometry-optin-r453-CPU-v1/CPU-result.json','d567329a41cc5c33e6f6bf39aef2b89209b3ef5e1756bebad4116538b2330c96');assert base['passed']and H(base['native'])==base['native_SHA256']=='fc3def9a539f3e9c9e1fc4c4ef2c2f0bbee03ddbe32300c689a8113310068056'
  assert H(P/'CPU_SOURCE_MANIFEST.json')=='f37bef41ef98b216c1c3b5a1a4a3015a0a161a64bd5d5842f96be5a96e0827cd'
  for p,h in json.loads((P/'CPU_SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert H(P/p)==h,p
  old179=receipt('/refactor/candidates/Kimi-long-agent32K-r179/CPU-build.json','37ac44052ffb3b0e43cf7e440178aad4096f09dcd97a8b12d6eca6fc65cd827d')
  ctx=Path('/refactor/incoming/Kimi-ctx2048-r123e/per-TU-original-context-r123e.json');assert H(ctx)=='07bb301b870f90e1ce9e8f771b2c73080937ee0d200ef753a7521fa81f922454';contexts=json.loads(ctx.read_bytes())
  sources=json.loads((D/'PARENT_SOURCES.json').read_bytes());originals={}
  for p,h in sources.items():assert H(p)==h,p;originals[p]={'exists':True,'rawtext':Path(p).read_bytes().decode()}
  from check_source_contract import check
  q['source_contract']=check(P/'native',originals)
  for p,h in base['all_linked_object_SHA256'].items():assert H(p)==h,p
  for v in base['actual_dependency_library_SHA256'].values():assert H(v['path'])==v['SHA256']
  # Four exact input-header consumers plus MoE. Header/head class source from
  # r453 stays unchanged, and its two compiled objects remain reused.
  units=[]
  for unit,name in [('pair_probe','pair_probe.cpp'),('allocate','kimi_compact_allocate_r4974.cpp'),('moe','kimi_moe_scale_exact_r201.cpp')]:
   cmd=next(c for c in base['commands']if '-c'in c and Path(c[c.index('-c')+1]).name==name);oldobj=str(Path('/refactor/candidates/Kimi-long-geometry-optin-r453-CPU-v1')/(unit+'.o'));units.append((unit,name,cmd,base['HOST'][unit]['all_new_MD'],str(P/'native'),oldobj))
  for unit,name in [('worker','worker.cpp'),('infer_engine','kimi_compact_engine_r4974.cpp')]:units.append((unit,name,contexts[unit]['argv'],old179['all_new_MD'][unit],'/refactor/incoming/Kimi-long-agent32K-r179/native','/refactor/candidates/Kimi-long-agent32K-r179/'+unit+'.o'))
  inputp='/refactor/incoming/Kimi-long-agent32K-r179/native/kimi_engine_input_r145.hpp';mds={**{k:v['all_new_MD']for k,v in base['HOST'].items()},**{k:v for k,v in old179['all_new_MD'].items()if k in ['worker','infer_engine','delta','trace']}}
  assert {k for k,v in mds.items()if inputp in v}=={'pair_probe','allocate','worker','infer_engine'}
  # New explicit Fill/submit declarations may add dependencies to allocator.
  # Every added external byte must already be in a successful current parent
  # MD; do not broadly ignore new headers or substitute arbitrary SDK bytes.
  known={}
  for md in mds.values():
   for p,h in md.items():
    c=canonical(p)
    if c in known:assert known[c]==h,c
    known[c]=h
  source_map={p.name:str(P/'native'/p.name)for p in (P/'native').iterdir()if p.is_file()}
  source_map.update({n:'/refactor/incoming/Kimi-long-agent32K-r179/native/'+n for n in ['worker.cpp','kimi_compact_engine_r4974.cpp','kimi_engine_input_r145.hpp']})
  extras={str(N/n)for n in ['ar_storage_r456.hpp','exact_mla_cache_r456.hpp']};changed={str(P/'native'/n)for n in ['pair_probe.cpp','kimi_compact_allocate_r4974.cpp','kimi_moe_scale_exact_r201.cpp']}|{inputp}
  objects={};details={}
  for unit,name,argv,oldmd,parent,oldobj in units:
   f=semantic(argv);assert Path(f[0]).name=='g++'and '-O2'in f and '-fPIC'in f and '-pthread'in f
   flags=[f[0],'-I'+str(N),'-I'+parent,'-I'+parent+'/engine','-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine',*f[1:]];out=B/(unit+'.o')
   run([*flags,'-MD','-MF',str(B/(unit+'.d')),'-c',str(N/name),'-o',str(out)],unit+'-HOST')
   run([*flags,'-M','-MF',str(B/(unit+'.fresh.d')),'-MT',str(out),str(N/name)],unit+'-fresh-M',30)
   actual=deps(B/(unit+'.d'));assert actual==deps(B/(unit+'.fresh.d'))
   oldcanon={canonical(p):h for p,h in oldmd.items()};normalized={}
   for p in actual:
    origin=source_map[Path(p).name]if Path(p).parent==N and p not in extras else p;normalized[canonical(origin)]=p
   real=set(normalized)-extras;added=real-set(oldcanon);missing=set(oldcanon)-real;assert not missing,(unit,'missing MD',sorted(missing))
   assert added<=set(known),(unit,'new unpinned dependency',sorted(added-set(known)))
   for p,v in normalized.items():
    if p in extras or p in changed or p==canonical(parent+'/'+name):continue
    assert H(v)==oldcanon.get(p,known.get(p)),(unit,'dependency SHA changed',p)
   objects[oldobj]=str(out);details[unit]=dict(source_SHA256=H(N/name),object_SHA256=H(out),all_new_MD={p:H(p)for p in sorted(actual)},fresh_MD_matches=True,added_external_dependencies_from_successful_parent_MD={p:known[p]for p in sorted(added)})
  link=next(c[:]for c in base['commands']if '-shared'in c);assert {p for p in link if p.endswith('.o')}==set(base['all_linked_object_SHA256'])
  link=[objects.get(p,p)for p in link];native=B/'libKimi_AR_persistent_storage_r456.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(B/'link.map')if p.startswith('-Wl,-Map')else p)for p in link]
  assert '-Wl,-Bsymbolic'in link and '-Wl,--no-undefined'in link;run(link,'Native-isolated-link')
  def symbols(s):return{w[-1]:w[-2]for line in s.splitlines()if len(w:=line.split())>=2}
  old=symbols(run(['nm','-D','--defined-only',base['native']],'old-defined',20));new=symbols(run(['nm','-D','--defined-only',str(native)],'new-defined',20));strong={p for p,t in old.items()if t.upper()in {'T','D','B','R','G','S','A'}};assert strong<=set(new)
  oldc={p for p in old if p.startswith('modern_')};newc={p for p in new if p.startswith('modern_')};expected={'modern_kimi_engine_create_ar_storage_r456','modern_kimi_engine_forward_ar_storage_r456','modern_kimi_ar_storage_capabilities_r456'};assert oldc<=newc and newc-oldc==expected
  ldd=run(['ldd','-r',str(native)],'ldd-r',45);assert 'not found'not in ldd and 'undefined symbol:'not in ldd;resolved={}
  for line in ldd.splitlines():
   w=line.split();p=w[2]if len(w)>2 and w[1]=='=>'else w[0]if w and w[0].startswith('/')else None
   if p and Path(p).name in base['actual_dependency_library_SHA256']:
    key=Path(p).name;assert H(p)==base['actual_dependency_library_SHA256'][key]['SHA256'];resolved[key]=dict(path=p,SHA256=H(p))
  assert set(resolved)==set(base['actual_dependency_library_SHA256'])
  from prepare_CPU_metadata_contract import prepare
  q['actual_template_recording_leaf_identity']=prepare(N/'kimi_engine_input_r145.hpp',B/'actual_freeze_input_recording.hpp')
  exe=B/'CPU-contract';run(['g++','-std=c++17','-O2','-I'+str(B),'-I'+str(D),'-I'+str(N),'-I/refactor/incoming/Kimi-long-agent32K-r179/native',str(D/'CPU_contract.cpp'),'-o',str(exe)],'pure-contract-compile',30);q['pure_address_contract']=json.loads(run([str(exe)],'pure-contract-run',10));assert q['pure_address_contract']['passed']
  q.update(completed=True,passed=True,native=str(native),native_SHA256=H(native),HOST=details,all_linked_object_SHA256={p:H(p)for p in link if p.endswith('.o')},unchanged_objects_SHA256={p:h for p,h in base['all_linked_object_SHA256'].items()if p not in objects},actual_dependency_library_SHA256=resolved,old_strong_exports_retained=len(strong),old_CAPI_retained=len(oldc),new_CAPI=sorted(expected),qualification='HOST/MD/link/pure address-only; original4GiB floor unchanged. Actual TP32 CCL/allocator pressure, constructor lifecycle, checkpoint math and fit UNKNOWN; no device qualification.')
 except BaseException as e:q['error']=repr(e);raise
 finally:q['elapsed_s']=time.monotonic()-start;save();print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','elapsed_s','phase','error','native_SHA256']}))
if __name__=='__main__':main()
