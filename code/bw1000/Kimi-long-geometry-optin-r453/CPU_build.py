"""Private five-HOST build, 28 exact objects reused; no Native execution.

Only pure/header CPU consumer executes. ldd-r resolves ELF dependencies without
calling the candidate's constructors. No ctypes/CDLL/import of model modules.
"""
from pathlib import Path
import ast,hashlib,json,os,shlex,subprocess,sys,time
D=Path(__file__).resolve().parent;N=D/'native';B=Path('/refactor/candidates/Kimi-long-geometry-optin-r453-CPU-v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def semantic(a):
 out=[a[0]];i=1
 while i<len(a):
  z=a[i]
  if z in ['-o','-MF','-MT','-MQ','-MJ']:i+=2;continue
  if z in ['-c','-MD','-MMD','-MP','-MG','-M','-MM','-shared']:i+=1;continue
  if z.startswith(('-MF','-MT','-MQ','-MJ','-Wl,','-L','-l'))or z.endswith(('.cpp','.cc','.cxx','.o','.so','.a')):i+=1;continue
  out.append(z);i+=1
 return out
def deps(p):return set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]))
def main():
 assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 assert 'torch'not in sys.modules and 'transformers'not in sys.modules and not os.environ.get('LD_PRELOAD')
 m=json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())
 for p,h in m['files_SHA256'].items():assert sha(D/p)==h,p
 assert not B.exists();B.mkdir();start=time.monotonic()
 q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Model=0,GPU_GO=False,Model_GO=False,Native_dlopen=False,Runtime_created=False,new_HOST_TUs=5,new_GPU_TUs=0,pure_contract_TUs=1,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=sha(D/'CPU_SOURCE_MANIFEST.json'),commands=[],logs={})
 def save():(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(a,label,cap=300):
  left=1740-(time.monotonic()-start);assert left>0
  q['phase']=label;q['commands'].append(a);save()
  try:p=subprocess.run(a,capture_output=True,text=True,timeout=min(left,cap))
  except subprocess.TimeoutExpired as e:
   log=B/(label+'.log');log.write_text((e.stdout or b'').decode(errors='replace')+(e.stderr or b'').decode(errors='replace'));q['logs'][label]=dict(timeout=True,SHA256=sha(log));raise
  log=B/(label+'.log');log.write_text(p.stdout+p.stderr);q['logs'][label]=dict(rc=p.returncode,SHA256=sha(log))
  if p.returncode:raise RuntimeError(label+' failed; '+str(log))
  return p.stdout
 def receipt(p,h):
  assert sha(p)==h,p;v=json.loads(Path(p).read_bytes());assert v['completed']and v['CPU_only'];return v
 try:
  for p in D.glob('*.py'):ast.parse(p.read_bytes())
  from check_source_contract import check,originals_remote
  q['source_contract']=check(originals_remote());save()
  base=receipt('/refactor/candidates/Kimi-append512-r247/CPU-build.json','21fc8ed69c6db1ec3a3c8925e61dc5f73112b5fea596a8caef977008da57046b')
  assert sha(base['native'])==base['native_SHA256']=='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
  old179=receipt('/refactor/candidates/Kimi-long-agent32K-r179/CPU-build.json','37ac44052ffb3b0e43cf7e440178aad4096f09dcd97a8b12d6eca6fc65cd827d')
  old201=receipt('/refactor/candidates/Kimi-long-scale-exact-r201/CPU-build.json','e04ddad0f5b9bc23ad4ef5ac24353dce20ac4726bb737fcf11553311d7738336')
  oldhead=receipt('/refactor/candidates/Kimi-shard-only-Model-r9183/CPU-build.json','d7ddf1c7d136a78c94ef9c302cd176d51f951c229e857e5d1be252c41b1bc556')
  for p,h in base['all_linked_object_SHA256'].items():assert sha(p)==h,p
  for v in base['actual_dependency_library_SHA256'].values():assert sha(v['path'])==v['SHA256']
  ctx=Path('/refactor/incoming/Kimi-ctx2048-r123e/per-TU-original-context-r123e.json');assert sha(ctx)=='07bb301b870f90e1ce9e8f771b2c73080937ee0d200ef753a7521fa81f922454';contexts=json.loads(ctx.read_bytes())
  pc=Path('/refactor/incoming/Kimi-long-scale-exact-r201/actual-context-pins-r201.json');assert sha(pc)=='fc173934f81578d9fe5883363112fea5a3126e59b871638a30875ebf3e9657f4';pins=json.loads(pc.read_bytes())
  oldmoe=receipt(pins['moe_receipt'],pins['moe_receipt_SHA256']);moecc=next(a for a in oldmoe['commands']if '-c'in a and a[a.index('-c')+1]==pins['qualified_moe_source'])
  hd='/refactor/incoming/Kimi-shard-only-Model-r9183'
  units=[('pair_probe','pair_probe.cpp',contexts['pair_probe']['argv'],base['all_new_MD']['pair_probe'],'/refactor/incoming/Kimi-append512-r247/native','/refactor/candidates/Kimi-append512-r247/pair_probe.o'),('allocate','kimi_compact_allocate_r4974.cpp',contexts['allocate']['argv'],old179['all_new_MD']['allocate'],'/refactor/incoming/Kimi-long-agent32K-r179/native','/refactor/candidates/Kimi-long-agent32K-r179/allocate.o'),('moe','kimi_moe_scale_exact_r201.cpp',moecc,old201['all_new_MD']['moe'],'/refactor/incoming/Kimi-long-scale-exact-r201','/refactor/candidates/Kimi-long-scale-exact-r201/moe.o')]
  for unit,name in [('head','kimi_shard_model_r9183.cpp'),('policy','kimi_shard_policy_r9183.cpp')]:
   a=next(c for c in oldhead['commands']if '-c'in c and Path(c[c.index('-c')+1]).name==name)
   units.append((unit,name,a,oldhead['new_complete_freshMD'][unit],hd,'/refactor/candidates/Kimi-shard-only-Model-r9183/'+unit+'.o'))
  # Actual successful MD sets of every later retained HOST rebuild; this catches
  # worker/delta/trace/infer_engine transitive inclusion of modified head/scale.
  knownmd={**{('r179/'+k):v for k,v in old179['all_new_MD'].items()if k not in ['pair_probe','allocate']},**{u:md for u,_,_,md,_,_ in units}}
  changed_headers={hd+'/'+n for n in ['kimi_shard_causal_lm_r9183.hpp','kimi_shard_head_r9183.hpp','kimi_shard_stats_r9183.hpp']}|{'/refactor/incoming/Kimi-long-scale-exact-r201/scale_exact_scope_r201.hpp'}
  includers={h:sorted(k for k,md in knownmd.items()if h in md)for h in changed_headers}
  assert all(set(v)<=set(u[0]for u in units)for v in includers.values()),includers
  assert set(oldhead['new_complete_freshMD'])=={'head','policy'}
  q['actual_known_MD_changed_header_includers']=includers
  objects={};details={};contract_flags=None
  for unit,name,argv,oldmd,parent,oldobj in units:
   flags=semantic(argv);assert Path(flags[0]).name=='g++'and '-fPIC'in flags and '-O2'in flags and '-pthread'in flags
   assert not any(z.endswith('.cu')or z in ['-xhip','--hip-link']for z in flags)
   prefix=['-I'+str(N),'-I'+parent,'-I'+parent+'/engine','-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine'];cmd=[flags[0],*prefix,*flags[1:]]
   if unit=='head':contract_flags=cmd
   out=B/(unit+'.o');run([*cmd,'-MD','-MF',str(B/(unit+'.d')),'-c',str(N/name),'-o',str(out)],unit+'-HOST')
   run([*cmd,'-M','-MF',str(B/(unit+'.fresh.d')),'-MT',str(out),str(N/name)],unit+'-fresh-M',60)
   actual=deps(B/(unit+'.d'));assert actual==deps(B/(unit+'.fresh.d'))
   extra={str(N/'long_geometry_r453.hpp')};normalized={}
   # Normalize all private file names to their exact original source locations.
   source_map={Path(p).name:p for p in json.loads((D/'DERIVATION.json').read_bytes())['original_raw_source_SHA256']}
   for p in actual:normalized[source_map[Path(p).name]if Path(p).parent==N and p not in extra else p]=p
   assert set(normalized)-extra==set(oldmd),(unit,'MD set changed',sorted(set(normalized)-set(oldmd)-extra),sorted(set(oldmd)-set(normalized)))
   changed={parent+'/'+name}|changed_headers
   for p,real in normalized.items():
    if p not in changed|extra:assert sha(real)==oldmd[p],(unit,'unchanged dependency bytes',p)
   assert sha(oldobj)==base['all_linked_object_SHA256'][oldobj]
   objects[oldobj]=str(out);details[unit]=dict(source_SHA256=sha(N/name),object_SHA256=sha(out),all_new_MD={p:sha(p)for p in sorted(actual)},fresh_MD_matches=True)
  # Old complete link order -> actual CPU622 object identity at each position.
  link=next(a[:]for a in oldhead['commands']if '-shared'in a);order=list(base['all_linked_object_SHA256']);cursor=0
  for i,p in enumerate(link):
   if p.endswith('.o'):
    assert Path(p).name==Path(order[cursor]).name or (Path(p).name,Path(order[cursor]).name)==('kimi_k3_allocate_cache.o','allocate.o')
    link[i]=order[cursor];cursor+=1
  assert cursor==len(order)==33 and len(objects)==5
  # Unrebuilt objects must not carry emitted references/definitions of changed
  # private stats or scale helper. Together with known MD this prevents a stale
  # inline variable/descriptor implementation from the actual linked graph.
  reuse={p:h for p,h in base['all_linked_object_SHA256'].items()if p not in objects}
  symtext=run(['nm','-C',*reuse],'reused-object-symbols',60)
  assert 'kimi_shard_r9183::'not in symtext and 'kimi_scale_exact_r201::'not in symtext
  q['reused_object_private_helper_symbol_absence']=True
  native=B/'libKimi_long_geometry_optin_r453.so';link=[objects.get(p,p)for p in link];link[link.index('-o')+1]=str(native)
  link=[('-Wl,-Map,'+str(B/'link.map')if p.startswith('-Wl,-Map')else p)for p in link]
  assert '-Wl,-Bsymbolic'in link and '-Wl,--no-undefined'in link
  run(link,'Native-isolated-link',150)
  def symbols(s):return{w[-1]:w[-2]for l in s.splitlines()if len(w:=l.split())>=2}
  old=symbols(run(['nm','-D','--defined-only',base['native']],'old-defined',45));new=symbols(run(['nm','-D','--defined-only',str(native)],'new-defined',45))
  strong={p:t for p,t in old.items()if t.upper()in {'T','D','B','R','G','S','A'}and t not in {'W','V','w','v'}}
  missing={p:t for p,t in strong.items()if p not in new};assert not missing,missing
  oldc={p for p in old if p.startswith('modern_')};newc={p for p in new if p.startswith('modern_')}
  expected={'modern_kimi_engine_create_long_geometry_r453','modern_kimi_engine_forward_long_geometry_r453','modern_kimi_long_geometry_capabilities_r453'}
  assert oldc<=newc and newc-oldc==expected
  ldd=run(['ldd','-r',str(native)],'ldd-r',90);assert 'not found'not in ldd and 'undefined symbol:'not in ldd
  actual={}
  for line in ldd.splitlines():
   w=line.split();p=(w[2]if len(w)>2 and w[1]=='=>'else w[0]if w and w[0].startswith('/')else None)
   if p and Path(p).name in base['actual_dependency_library_SHA256']:
    key=Path(p).name;assert sha(p)==base['actual_dependency_library_SHA256'][key]['SHA256'];actual[key]=dict(path=p,SHA256=sha(p))
  assert set(actual)==set(base['actual_dependency_library_SHA256'])
  exe=B/'CPU-contract';run([*contract_flags,'-I'+str(D),str(D/'CPU_contract.cpp'),'-o',str(exe)],'pure-header-contract-compile',120)
  q['actual_header_CPU_contract']=json.loads(run([str(exe)],'pure-header-contract-run',15));assert q['actual_header_CPU_contract']['passed']
  q.update(completed=True,passed=True,Native_linked=True,native=str(native),native_SHA256=sha(native),HOST=details,all_linked_object_SHA256={p:sha(p)for p in link if p.endswith('.o')},unchanged_objects_SHA256=reuse,actual_dependency_library_SHA256=actual,old_strong_exports_retained=len(strong),old_CAPI_retained=len(oldc),new_CAPI=sorted(expected),qualification='Actual HOST compilation/static ELF dependency and pure CPU geometry only. No Runtime/Native constructor/loader/32owner lifecycle/device math/fit/quality/performance qualification.')
 except BaseException as e:q['error']=repr(e);raise
 finally:q['elapsed_s']=time.monotonic()-start;save();print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','elapsed_s','phase','error','native_SHA256']}))
if __name__=='__main__':main()
