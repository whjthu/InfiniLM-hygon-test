"""Three changed HOST TUs; original 30 objects and exact dynamic dependencies reused."""
from pathlib import Path
import ast,hashlib,json,os,shlex,subprocess,sys,time
D=Path(__file__).resolve().parent;N=D/'native';R=Path('/refactor')
B=R/'candidates/Kimi-TP8PP4-ordinary-r410-CPU-v1'
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
 q={'completed':False,'passed':False,'CPU_only':True,'GPU':0,'Model':0,'Native_dlopen':False,
    'Runtime_created':False,'GPU_GO':False,'new_HOST_TUs':3,'new_GPU_TUs':0,
    'actual_CPU':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
    'source_manifest_SHA256':sha(D/'CPU_SOURCE_MANIFEST.json'),'logs':{}}
 def run(a,label,cap=90):
  left=270-(time.monotonic()-start);assert left>0
  p=subprocess.run(a,capture_output=True,text=True,timeout=min(left,cap));log=B/(label+'.log');log.write_text(p.stdout+p.stderr)
  q['logs'][label]={'rc':p.returncode,'SHA256':sha(log)}
  if p.returncode:raise RuntimeError(label+' failed; see '+str(log))
  return p.stdout
 def receipt(p,h):
  assert sha(p)==h,p;v=json.loads(Path(p).read_bytes());assert v['completed']and v['CPU_only'];return v
 try:
  for p in D.glob('*.py'):ast.parse(p.read_bytes())
  base=receipt('/refactor/candidates/Kimi-append512-r247/CPU-build.json','21fc8ed69c6db1ec3a3c8925e61dc5f73112b5fea596a8caef977008da57046b')
  assert base['native_SHA256']=='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'and sha(base['native'])==base['native_SHA256']
  old179=receipt('/refactor/candidates/Kimi-long-agent32K-r179/CPU-build.json','37ac44052ffb3b0e43cf7e440178aad4096f09dcd97a8b12d6eca6fc65cd827d')
  old201=receipt('/refactor/candidates/Kimi-long-scale-exact-r201/CPU-build.json','e04ddad0f5b9bc23ad4ef5ac24353dce20ac4726bb737fcf11553311d7738336')
  for p,h in base['all_linked_object_SHA256'].items():assert sha(p)==h,p
  for _,v in base['actual_dependency_library_SHA256'].items():assert sha(v['path'])==v['SHA256'],v['path']
  ctx=Path('/refactor/incoming/Kimi-ctx2048-r123e/per-TU-original-context-r123e.json')
  assert sha(ctx)=='07bb301b870f90e1ce9e8f771b2c73080937ee0d200ef753a7521fa81f922454';contexts=json.loads(ctx.read_bytes())
  p201=Path('/refactor/incoming/Kimi-long-scale-exact-r201/actual-context-pins-r201.json')
  assert sha(p201)=='fc173934f81578d9fe5883363112fea5a3126e59b871638a30875ebf3e9657f4';pins=json.loads(p201.read_bytes())
  oldmoe=receipt(pins['moe_receipt'],pins['moe_receipt_SHA256'])
  moecc=next(a for a in oldmoe['commands']if '-c'in a and a[a.index('-c')+1]==pins['qualified_moe_source'])
  units=[('pair_probe','pair_probe.cpp',contexts['pair_probe']['argv'],base['all_new_MD']['pair_probe'],'/refactor/incoming/Kimi-append512-r247/native','/refactor/candidates/Kimi-append512-r247/pair_probe.o'),
         ('allocate','kimi_compact_allocate_r4974.cpp',contexts['allocate']['argv'],old179['all_new_MD']['allocate'],'/refactor/incoming/Kimi-long-agent32K-r179/native','/refactor/candidates/Kimi-long-agent32K-r179/allocate.o'),
         ('moe','kimi_moe_scale_exact_r201.cpp',moecc,old201['all_new_MD']['moe'],'/refactor/incoming/Kimi-long-scale-exact-r201','/refactor/candidates/Kimi-long-scale-exact-r201/moe.o')]
  objects={};details={}
  for unit,name,argv,oldmd,parent,oldobj in units:
   flags=semantic(argv);assert Path(flags[0]).name=='g++'and '-fPIC'in flags and '-O2'in flags and '-pthread'in flags
   assert not any(z.endswith('.cu')or z in ['-xhip','--hip-link']for z in flags)
   prefix=['-I'+str(N),'-I'+parent,'-I'+parent+'/engine','-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine']
   out=B/(unit+'.o');cmd=[flags[0],*prefix,*flags[1:]]
   run([*cmd,'-MD','-MF',str(B/(unit+'.d')),'-c',str(N/name),'-o',str(out)],unit+'-HOST')
   run([*cmd,'-M','-MF',str(B/(unit+'.fresh.d')),'-MT',str(out),str(N/name)],unit+'-fresh-M',30)
   actual=deps(B/(unit+'.d'));assert actual==deps(B/(unit+'.fresh.d'))
   oldsrc=parent+'/'+name;changed={oldsrc}
   if unit=='moe':changed.add(parent+'/scale_exact_scope_r201.hpp')
   newheaders={str(N/'topology_r410.hpp')}
   normalized={}
   for p in actual:
    normalized[parent+'/'+Path(p).name if Path(p).parent==N and p not in newheaders else p]=p
   assert set(normalized)-newheaders==set(oldmd),(unit,'dependency set changed',sorted(set(normalized)-set(oldmd)-newheaders),sorted(set(oldmd)-set(normalized)))
   for p,actualp in normalized.items():
    if p not in changed|newheaders:assert sha(actualp)==oldmd[p],(unit,'dependency bytes changed',p)
   assert newheaders<=actual and sha(oldobj)==base['all_linked_object_SHA256'][oldobj]
   objects[oldobj]=str(out);details[unit]={'source_SHA256':sha(N/name),'object_SHA256':sha(out),'all_new_MD':{p:sha(p)for p in sorted(actual)},'fresh_MD_matches':True}
  # Existing full original link command; replace only the three known current objects.
  lb=receipt('/refactor/candidates/Kimi-W2-pair-r9280/CPU-build.json','e86d3428b6134873f5525cbfd0409168593ec1d17c2fa3b48bfcebfa46ebf86e')
  link=next(a[:]for a in lb['commands']if '-shared'in a)
  oldobjs=[x for x in link if x.endswith('.o')];assert len(oldobjs)==len(base['all_linked_object_SHA256'])
  # Earlier qualified link replaces six context TUs and the exact scale TU.
  suffixes={'pair_probe.o':'pair_probe','worker.o':'worker','delta.o':'delta','trace.o':'trace','infer_engine.o':'infer_engine','kimi_k3_allocate_cache.o':'allocate'}
  for i,x in enumerate(link):
   if not x.endswith('.o'):continue
   if x in base['all_linked_object_SHA256']:continue
   name=Path(x).name
   if name in suffixes:link[i]='/refactor/candidates/Kimi-long-agent32K-r179/'+suffixes[name]+'.o'
   elif name=='moe.o':link[i]='/refactor/candidates/Kimi-long-scale-exact-r201/moe.o'
  # r247 then replaces the r179 pair object with the actual append512 object.
  link=[('/refactor/candidates/Kimi-append512-r247/pair_probe.o'if x=='/refactor/candidates/Kimi-long-agent32K-r179/pair_probe.o'else x)for x in link]
  assert {x for x in link if x.endswith('.o')}==set(base['all_linked_object_SHA256'])
  native=B/'libKimi_TP8PP4_ordinary_r410.so';link[link.index('-o')+1]=str(native)
  for i,x in enumerate(link):
   if x in objects:link[i]=objects[x]
   if x.startswith('-Wl,-Map'):link[i]='-Wl,-Map,'+str(B/'link.map')
  run(link,'Native-link',90)
  capi=lambda s:{l.split()[-1]for l in s.splitlines()if l.split()and l.split()[-1].startswith('modern_')}
  oldc=capi(run(['nm','-D','--defined-only',base['native']],'old-CAPI',20));newc=capi(run(['nm','-D','--defined-only',str(native)],'new-CAPI',20))
  expected={'modern_kimi_engine_create_ordinary_tp8_pp4_r410','modern_kimi_engine_forward_ordinary_r410','modern_kimi_ordinary_capabilities_r410'}
  assert oldc<=newc and newc-oldc==expected
  ldd=run(['ldd','-r',str(native)],'ldd-r',45);assert 'not found'not in ldd and 'undefined symbol:'not in ldd
  actual={}
  for l in ldd.splitlines():
   w=l.split()
   if len(w)>=3 and w[1]=='=>'and w[0]in base['actual_dependency_library_SHA256']:
    assert sha(w[2])==base['actual_dependency_library_SHA256'][w[0]]['SHA256'];actual[w[0]]={'path':w[2],'SHA256':sha(w[2])}
  assert set(actual)==set(base['actual_dependency_library_SHA256'])
  exe=B/'CPU-contract';run(['g++','-std=c++17','-O2','-I'+str(N),str(D/'CPU_contract.cpp'),'-o',str(exe)],'contract-compile',30)
  q['geometry_contract']=json.loads(run([str(exe)],'contract-run',10))
  q.update(completed=True,passed=True,Native_linked=True,native=str(native),native_SHA256=sha(native),
           HOST=details,all_linked_object_SHA256={p:sha(p)for p in link if p.endswith('.o')},
           unchanged_objects_SHA256={p:h for p,h in base['all_linked_object_SHA256'].items()if p not in objects},
           actual_dependency_library_SHA256=actual,old_CAPI_retained=len(oldc),new_CAPI=sorted(expected),
           qualification='Actual HOST/link/dependency and pure geometry only; no device constructor/loader/forward/quality/physical fit proof')
 except BaseException as e:q['error']=repr(e);raise
 finally:
  q['elapsed_s']=time.monotonic()-start;(B/'CPU-result.json').write_bytes((json.dumps(q,indent=2)+'\n').encode())
  print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','elapsed_s','error','native_SHA256']}))
if __name__=='__main__':main()
