"""Six HOST TUs against locked original per-TU compiler contexts. No HIP
compilation, Native dlopen, Runtime/model/GPU execution or job submission.
Original compiler argv are read only on the cluster and are never printed.
"""
import argparse,ast,hashlib,json,os,pathlib,shlex,subprocess,sys,time,traceback
D=pathlib.Path(__file__).resolve().parent;N=D/'native';R=pathlib.Path('/refactor');B=R/'candidates/Kimi-long-agent32K-r179';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
BASES={'pair_probe':'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/pair_probe.o','worker':'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/worker.o','delta':'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/delta.o','trace':'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/trace.o','infer_engine':'/refactor/candidates/kimi-compact-cache-r4974/objects/infer_engine.o','allocate':'/refactor/candidates/kimi-compact-cache-r4974/objects/kimi_k3_allocate_cache.o'}
SOURCES={'pair_probe':'pair_probe.cpp','worker':'worker.cpp','delta':'delta.cpp','trace':'kimi_live_trace_r9046.cpp','infer_engine':'kimi_compact_engine_r4974.cpp','allocate':'kimi_compact_allocate_r4974.cpp'}
def container_path(p):return str(p).replace('/public/home/aclsylqidf/hygon-inference-refactor-20261014','/refactor',1)
def semantic(c):
 result=[c[0]];i=1
 while i<len(c):
  x=c[i]
  if x in ['-o','-MF','-MT','-MQ','-MJ']:i+=2;continue
  if x in ['-c','-MD','-MMD','-MP','-MG','-M','-MM','-shared']:i+=1;continue
  if x.startswith(('-MF','-MT','-MQ','-MJ','-Wl,','-L','-l')):i+=1;continue
  if x.endswith(('.cpp','.cc','.cxx','.o','.so','.a')):i+=1;continue
  result.append(x);i+=1
 return result

def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--cpu-build-authorized',action='store_true');x=a.parse_args();assert x.cpu_build_authorized
 mf=D/'source-manifest-r179.json';assert sha(mf)==x.manifest_sha;m=json.loads(mf.read_bytes())
 for name,h in m['source_SHA256'].items():assert sha(D/name)==h,name
 for p in D.rglob('*.py'):ast.parse(p.read_bytes())
 assert all(os.environ.get(k)==' ' .strip()for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 assert 'torch'not in sys.modules and 'transformers'not in sys.modules
 assert not B.exists();B.mkdir();start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'GPU_execution':False,'Native_dlopen':False,'Runtime_created':False,'new_HIP_TUs':0,'new_HOST_TUs':6,'GPU_GO':False,'actual_step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),'source_manifest_SHA256':x.manifest_sha,'compiler_context_argv_exported':False}
 def run(c,label,seconds=240):
  z=subprocess.run(c,text=True,capture_output=True,timeout=seconds);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,('CPU_HOST_STEP_FAILED',label,z.returncode);return z.stdout
 try:
  binding=json.loads((D/'actual-baseline-binding-r179.json').read_bytes());receipt=pathlib.Path(binding['source_receipt_path']);assert sha(receipt)==binding['source_receipt_SHA256'];observed=json.loads(receipt.read_bytes());assert observed['all_required_source_matches']
  for row in binding['rows']:
   expected=row['expected_LF_SHA256'];assert any(hashlib.sha256(pathlib.Path(container_path(p['path'])).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected for p in row['matches']),row['relative']
  contexts_path=pathlib.Path(binding['contexts_path']);assert sha(contexts_path)==binding['contexts_SHA256'];contexts=json.loads(contexts_path.read_bytes());assert set(contexts)==set(BASES)
  expected=json.loads((D/'base-object-pins-r179.json').read_bytes());base=json.loads((R/'candidates/Kimi-W2-pair-r9280/CPU-build.json').read_bytes());assert base['completed']and base['native_SHA256']==expected['native_SHA256']=='3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd';assert sha(base['native'])==base['native_SHA256'];assert base['all_linked_objects']==expected['all_linked_object_SHA256']
  for p,h in expected['all_linked_object_SHA256'].items():assert sha(p)==h,p
  for header in ['engine/infer_engine.hpp','engine/rank_worker.hpp','rank_worker.hpp']:assert (N/header).read_bytes()==(D/'original'/header).read_bytes(),header
  q['public_engine_worker_class_headers_unchanged']=True;q['opaque_Context_layout_only_pair_TU']=True
  prefix=['-I'+str(N),'-I'+str(N/'engine'),'-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine']
  commands={}
  for label in SOURCES:
   ctx=contexts[label];assert sha(container_path(ctx['receipt_path']))==ctx['receipt_SHA256'];assert sha(BASES[label])==ctx['object_SHA256'];flags=semantic(ctx['argv']);assert pathlib.Path(flags[0]).name in ['g++','g++.exe']and '-O2'in flags and '-fPIC'in flags and '-pthread'in flags
   if 'compile_flags_manifest_SHA256'in ctx:assert sha('/refactor/candidates/kimi-host-object-cache-5f91ad7-r4462/manifest.json')==ctx['compile_flags_manifest_SHA256']
   assert not any(z.endswith('.cu')or z in ['-xhip','--hip-link']for z in flags)
   commands[label]=[flags[0],*prefix,*flags[1:],'-MD','-MF',str(B/(label+'.d')),'-c',str(N/SOURCES[label]),'-o',str(B/(label+'.o'))]
   run(commands[label],label)
  # stdlib-only State/payload/table contracts, plus original legacy regression.
  for name,src in [('new',D/'CPU_contract_r179.cpp'),('old',D/'original/CPU_contract.cpp')]:
   exe=B/('CPU-'+name);run(['g++','-std=c++17','-O2','-I'+str(N),str(src),'-o',str(exe)],name+'-contract-compile',40);q[name+'_pure_contract']=json.loads(run([str(exe)],name+'-contract-run',20))
  q['runtime_transaction_contract']=json.loads(run([sys.executable,'-B',str(D/'CPU_runtime_contract_r179.py')],'runtime-contract',30))
  required={'pair_probe':['kimi_engine_input_r145.hpp','kimi_pair_state_r9006.hpp','batch_payload_r9122c.hpp'],'worker':['kimi_engine_input_r145.hpp','batch_payload_r9122c.hpp','batch_state_r9122.hpp'],'delta':['batch_state_r9122.hpp'],'trace':['batch_live_taps_r9122.hpp'],'infer_engine':['kimi_engine_input_r145.hpp','engine/infer_engine.hpp'],'allocate':['kimi_engine_input_r145.hpp','long_agent_geometry_r179.hpp']}
  md={};modified={p.name for p in N.rglob('*.hpp')if not(D/'original'/p.relative_to(N)).exists()or p.read_bytes()!=(D/'original'/p.relative_to(N)).read_bytes()}
  for label in SOURCES:
   path=B/(label+'.d');deps=shlex.split(path.read_text().replace('\\\n',' ').split(':',1)[1]);md[label]={p:sha(p)for p in deps if pathlib.Path(p).is_file()}
   for name in required[label]:assert md[label].get(str(N/name))==sha(N/name),(label,'candidate_header_not_bound',name)
   for name in modified:
    candidates=[p for p in md[label]if pathlib.Path(p).name==name];assert all(pathlib.Path(p).resolve().is_relative_to(N.resolve())for p in candidates),(label,'stale_modified_header',name)
   fresh=B/(label+'.fresh-M.d');run([*semantic(commands[label]),'-M','-MF',str(fresh),'-MT',str(B/(label+'.o')),str(N/SOURCES[label])],label+'-fresh-M',60)
   freshdeps=shlex.split(fresh.read_text().replace('\\\n',' ').split(':',1)[1]);assert set(deps)==set(freshdeps),(label,'effective_context_dependency_mismatch')
  # Link exactly the qualified base, replacing all six affected HOST objects.
  link=next(c[:]for c in base['commands']if '-shared'in c and '-o'in c and c[c.index('-o')+1]==base['native']);native=B/'libKimi_long_agent32K_r179.so'
  for label,path in BASES.items():assert link.count(path)==1;link[link.index(path)]=str(B/(label+'.o'))
  link[link.index('-o')+1]=str(native)
  for i,z in enumerate(link):
   if z.startswith('-Wl,-Map'):link[i]='-Wl,-Map,'+str(B/'link.map')
  run(link,'native-link',150);newobjs={p:sha(p)for p in link if p.endswith('.o')};unchanged={p:h for p,h in expected['all_linked_object_SHA256'].items()if p not in BASES.values()};assert all(newobjs.get(p)==h for p,h in unchanged.items())
  defined=run(['nm','-D','--defined-only',str(native)],'nm-defined',30);olddefs=subprocess.check_output(['nm','-D','--defined-only',base['native']],text=True,timeout=30);old_CAPI={line.split()[-1]for line in olddefs.splitlines()if len(line.split())>=3 and line.split()[-1].startswith('modern_')};new_CAPI={line.split()[-1]for line in defined.splitlines()if len(line.split())>=3 and line.split()[-1].startswith('modern_')};assert old_CAPI<=new_CAPI
  for symbol in ['modern_kimi_engine_create_long_agent_r179','modern_kimi_long_agent_capabilities_r179']:assert any(line.split()[-2:] == ['T',symbol]for line in defined.splitlines()if len(line.split())>=3)
  ldd=run(['ldd','-r',str(native)],'ldd-relocations',60);assert 'not found'not in ldd and 'undefined symbol:'not in ldd
  linked_libraries={}
  for line in ldd.splitlines():
   words=line.split()
   if len(words)>=3 and words[0]in expected['loaded_library_SHA256']and words[1]=='=>':
    assert sha(words[2])==expected['loaded_library_SHA256'][words[0]],words[0];linked_libraries[words[0]]={'path':words[2],'SHA256':sha(words[2])}
  assert set(linked_libraries)==set(expected['loaded_library_SHA256'])
  q['actual_dependency_library_SHA256']=linked_libraries
  profiles=B/'regular-agent-profiles-r179'
  run([sys.executable,'-B',str(D/'CPU_prepare_profiles_r179.py'),'--model','/shared/Kimi-K3','--output-dir',str(profiles)],'official-profiles',120)
  profile_receipt=profiles/'CPU-profile-receipt-r179.json';q['official_profile_receipt_path']=str(profile_receipt);q['official_profile_receipt_SHA256']=sha(profile_receipt);q['regular_profiles_CPU_only']=json.loads(profile_receipt.read_bytes())
  q.update(completed=True,native=str(native),native_SHA256=sha(native),all_linked_object_SHA256=newobjs,unchanged_objects_SHA256=unchanged,all_new_MD=md,original_context_path=str(contexts_path),original_context_SHA256=sha(contexts_path),old_CAPI_exports_preserved=len(old_CAPI),source_SHA256={p.relative_to(D).as_posix():sha(p)for p in sorted(N.rglob('*'))if p.is_file()},effective_compile_contexts_preserved=True,fresh_MD_dependency_sets_match=True,new_entry_context=32768,new_entry_logical=1,new_entry_physical=2,new_entry_state_pool=3,new_entry_KV_blocks=260,full_Model_peak='UNKNOWN',GPU_numeric_qualification='NOT_RUN',elapsed_s=time.monotonic()-start)
 except BaseException as error:q.update(error_type=type(error).__name__,error=('bounded subprocess timeout; original compiler argv withheld'if isinstance(error,subprocess.TimeoutExpired)else str(error)),elapsed_s=time.monotonic()-start)
 (B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','Native_dlopen','new_HOST_TUs','new_HIP_TUs','actual_step','native_SHA256','elapsed_s','error_type','error']}));assert q['completed']
if __name__=='__main__':main()
