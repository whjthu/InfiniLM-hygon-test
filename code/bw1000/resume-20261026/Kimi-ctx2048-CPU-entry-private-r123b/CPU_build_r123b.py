"""Authorizednextphase: sixHOSTobjects only; no GPU/Runtimecreate/dlopen."""
import pathlib,json,subprocess,hashlib,os,time,shlex,traceback,re
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;N=D/'native';B=R/'candidates/Kimi-ctx2048-r123b';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert os.environ.get('KIMI_CTX2048_HOST_BUILD_R123')=='1','HOSTbuild defaultOFF; rootcompile authorizationrequired';assert not B.exists();B.mkdir();q={'completed':False,'CPU_only':True,'GPU':False,'GPU_execution':False,'dlopen':False,'new_GPU_TUs':0,'candidate_HOST_TUs':6,'commands':[]};start=time.monotonic()
basepaths={
 'pair_probe':'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/pair_probe.o',
 'worker':'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/worker.o',
 'delta':'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/delta.o',
 'trace':'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/trace.o',
 'infer_engine':'/refactor/candidates/kimi-compact-cache-r4974/objects/infer_engine.o',
 'allocate':'/refactor/candidates/kimi-compact-cache-r4974/objects/kimi_k3_allocate_cache.o'}
sources={'pair_probe':'pair_probe.cpp','worker':'worker.cpp','delta':'delta.cpp','trace':'kimi_live_trace_r9046.cpp','infer_engine':'kimi_compact_engine_r4974.cpp','allocate':'kimi_compact_allocate_r4974.cpp'}
def run(c,label,timeout=240):
 q['commands'].append(c);z=subprocess.run(c,text=True,capture_output=True,timeout=timeout);(B/(label+'.log')).write_bytes((z.stdout+z.stderr).encode());assert z.returncode==0,(label,z.stderr[-12000:]);return z
try:
 pins=json.loads((D/'CPU-stage-file-manifest-r123b.json').read_bytes())
 for name,h in pins['files_SHA256'].items():assert sha(D/name)==h,name
 base=json.loads((R/'candidates/Kimi-W2-pair-r9280/CPU-build.json').read_bytes());assert base['completed']and sha(base['native'])==base['native_SHA256']=='3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd'
 original=base['all_linked_objects'];expected=json.loads((D/'base-object-pins-r123.json').read_bytes());assert original==expected['all_linked_object_SHA256']
 for p,h in original.items():assert sha(p)==h,p
 assert sha('/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib/libinfinilm_runtime.so')=='04a6a1a86c97d176e4304b9bb4c1d6fcb8550784beaa47e0f5841fb7603ffabf'
 assert 'torch'not in __import__('sys').modules
 import importlib.util,sys;torch_spec=importlib.util.find_spec('torch');assert torch_spec and torch_spec.origin;q['Torch_lib_path']=str(pathlib.Path(torch_spec.origin).parent/'lib');assert pathlib.Path(q['Torch_lib_path']).is_dir();assert 'torch'not in sys.modules
 q.update(actual_step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),hostname=__import__('socket').gethostname(),affinity_count=len(os.sched_getaffinity(0)))
 assert q['affinity_count']>=8 and all(os.environ.get(k)==' ' .strip()for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 contexts=json.loads((D/'per-TU-original-context-r123b.json').read_bytes());prefix=['-I'+str(N),'-I'+str(N/'engine'),'-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine']
 def semantic(c):
  # Scan ENTIRE argv, never truncate at dependency option. Preserve ordering/all compile semantics.
  result=[c[0]];i=1
  while i<len(c):
   x=c[i]
   if x in ['-o','-MF','-MT','-MQ','-MJ']:i+=2;continue
   if x in ['-c','-MD','-MMD','-MP','-MG','-M','-MM','-shared']:i+=1;continue
   if x.startswith(('-MF','-MT','-MQ','-MJ','-Wl,','-L','-l')):i+=1;continue
   if x.endswith(('.cpp','.cc','.cxx','.o','.so','.a')):i+=1;continue
   result.append(x);i+=1
  return result
 def command_for(label):
  ctx=contexts[label];assert sha(ctx['receipt_path'])==ctx['receipt_SHA256'];
  if 'compile_flags_manifest_SHA256'in ctx:assert sha('/refactor/candidates/kimi-host-object-cache-5f91ad7-r4462/manifest.json')==ctx['compile_flags_manifest_SHA256']
  assert sha(basepaths[label])==ctx['object_SHA256'];c=semantic(ctx['argv']);assert 'g++'in pathlib.Path(c[0]).name and '-O2'in c and '-fPIC'in c and '-pthread'in c
  return [c[0],*prefix,*c[1:],'-MD','-MF',str(B/(label+'.d')),'-c',str(N/sources[label]),'-o',str(B/(label+'.o'))]
 q['per_TU_context']=contexts;q['effective_compile_argv']={k:command_for(k)for k in sources}
 for label in sources:run(command_for(label),label)
 # CPUpureState/payloadregression; noRuntimeSDKexecution.
 z=run(['g++','-std=c++17','-O2','-I'+str(N),str(D/'CPU_contract_r123.cpp'),'-o',str(B/'CPU_contract')],'purecompile',40);z=run([str(B/'CPU_contract')],'purecontract',15);q['pure_contract']=json.loads(z.stdout)
 md={}
 required={'pair_probe':['kimi_engine_input_r145.hpp','kimi_pair_state_r9006.hpp','batch_payload_r9122c.hpp'],'worker':['kimi_engine_input_r145.hpp','batch_payload_r9122c.hpp','batch_state_r9122.hpp'],'delta':['batch_state_r9122.hpp'],'trace':['batch_live_taps_r9122.hpp'],'infer_engine':['kimi_engine_input_r145.hpp','engine/infer_engine.hpp'],'allocate':['kimi_engine_input_r145.hpp']}
 for label in sources:
  paths=shlex.split((B/(label+'.d')).read_text().replace('\\\n',' ').split(':',1)[1]);md[label]={p:sha(p)for p in paths if pathlib.Path(p).is_file()}
  fresh=B/(label+'.fresh-M.d');run([*semantic(command_for(label)),'-M','-MF',str(fresh),'-MT',str(B/(label+'.o')),str(N/sources[label])],label+'-fresh-M');freshpaths=shlex.split(fresh.read_text().replace('\\\n',' ').split(':',1)[1]);assert set(freshpaths)==set(paths),(label,'fresh-M effective-context mismatch')
  for name in required[label]:assert md[label].get(str(N/name))==sha(N/name),(label,'privateheadernotbound',name)
  for name in ['kimi_engine_input_r145.hpp','batch_payload_r9122c.hpp','batch_state_r9122.hpp','batch_live_taps_r9122.hpp','kimi_pair_state_r9006.hpp']:
   conflicts=[p for p in md[label]if pathlib.Path(p).name==name and p!=str(N/name)];assert not conflicts,(label,'staleprivateheader',conflicts)
 # Exactpublicengine/workerclassheaders preserved; compileBSSsizeof in actualsameHOSTenvironment.
 worker_cmd=command_for('worker');flags=semantic(worker_cmd);layout=B/'layout.cpp';layout.write_bytes(('#include "'+str(N/'engine/rank_worker.hpp')+'"\nextern "C" {char abi_worker[sizeof(infinilm::engine::RankWorker)];char abi_input[sizeof(infinilm::engine::RankWorker::Input)];char abi_output[sizeof(infinilm::engine::RankWorker::Output)];char abi_alignment[alignof(infinilm::engine::RankWorker)];}\n').encode());run([*flags,'-c',str(layout),'-o',str(B/'layout.o')],'layout',120);text=subprocess.check_output(['nm','-S',str(B/'layout.o')],text=True);q['ABI_layout']={x.split()[-1]:int(x.split()[1],16)for x in text.splitlines()if' abi_'in x};assert q['ABI_layout']=={'abi_worker':6912,'abi_input':752,'abi_output':128,'abi_alignment':8},q['ABI_layout']
 link=next(c[:]for c in base['commands']if'-o'in c and c[c.index('-o')+1]==base['native']);new=B/'libKimi_ctx2048_r123.so'
 for label,old in basepaths.items():assert link.count(old)==1;link[link.index(old)]=str(B/(label+'.o'))
 link[link.index('-o')+1]=str(new)
 for i,x in enumerate(link):
  if x.startswith('-Wl,-Map'):link[i]='-Wl,-Map,'+str(B/'link.map')
 run(link,'link',150);newobjects={p:sha(p)for p in link if p.endswith('.o')};unchanged={p:h for p,h in original.items()if p not in basepaths.values()};assert all(newobjects.get(p)==h for p,h in unchanged.items())
 defs=subprocess.check_output(['nm','-D','--defined-only',str(new)],text=True);olddefs=subprocess.check_output(['nm','-D','--defined-only',base['native']],text=True);modern={x.split()[-1]for x in olddefs.splitlines()if len(x.split())>=3 and x.split()[-1].startswith('modern_')};assert all(s in defs for s in modern)and'modern_kimi_engine_create_ctx2048_r123'in defs
 env=dict(os.environ,LD_LIBRARY_PATH=base['LD_LIBRARY_PATH']);dep=subprocess.run(['ldd','-r',str(new)],text=True,capture_output=True,env=env,timeout=60);assert dep.returncode==0 and'not found'not in dep.stdout+dep.stderr and'undefined symbol:'not in dep.stdout+dep.stderr
 q.update(completed=True,LD_LIBRARY_PATH=base['LD_LIBRARY_PATH'],native=str(new),native_SHA256=sha(new),all_linked_object_SHA256=newobjects,unchanged_object_SHA256=unchanged,actual_new_MD=md,replaced_objects=basepaths,ldd_relocations=dep.stdout+dep.stderr,source_SHA256=pins['files_SHA256'],original_native_SHA256=base['native_SHA256'],elapsed_s=time.monotonic()-start)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-start)
(B/'CPU-build.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','native_SHA256','elapsed_s','error']}));assert q['completed']
# Generatedmanifest is a distinct buildartifact; sourcePENDINGmanifest stays immutable.
m=json.loads((D/'runtime/model-manifest-r9183.json').read_bytes());m['native_DSO_SHA256']=q['native_SHA256'];m['native_receipt']=str(B/'CPU-build.json');m['resource_status']='HOST_BUILD_PASSED_ACTUAL_CPU_IMPORT_GPU_QUAL_PENDING';(B/'model-manifest-qualified-r123.json').write_bytes((json.dumps(m,indent=2)+'\n').encode())
