"""One pure HOST math consumer; link-only reuse of all three production objects."""
from pathlib import Path
import json, hashlib, subprocess, os, shlex, time, traceback, sys
R=Path('/refactor');S=Path(__file__).resolve().parent
B=R/'candidates/Deep-learned-router-suffix-r419-cpu-v6';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,CPU_only=True,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],publication_commit=os.environ['DEEP419_PUBLICATION_COMMIT'],commands=[],Native_dlopen=False,Runtime_created=False,GPU_initialized=False,weights_payload_opened=False,effective_LD_PRELOAD=os.environ.get('LD_PRELOAD'),actual_SDK_metadata_executed=False,SDK_metadata_deferred_to_real_device_context=True)
start=time.monotonic()
def run(args,label,timeout=100):
 q['commands'].append(args);p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 (B/(label+'.log')).write_text(p.stdout+p.stderr)
 assert p.returncode==0,(label,p.stderr[-7000:],p.stdout[-2000:]);return p.stdout
def deps(text):return set(shlex.split(text.replace(chr(92)+chr(10),' ').split(':',1)[1]))
try:
 mp=S/'SOURCE_MANIFEST.json';assert sha(mp)==os.environ['DEEP419_CPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 assert not q['effective_LD_PRELOAD'],'Unexpected preload: do not bypass an initializer with environment hacks'
 pin=json.loads((S/'PARENT_PINS.json').read_bytes());assert sha(pin['current_976f_CPU_receipt'])==pin['current_976f_CPU_receipt_SHA256'];native=json.loads(Path(pin['current_976f_CPU_receipt']).read_bytes())
 assert native['passed']and len(native['all_objects_SHA256'])==49 and all(sha(p)==h for p,h in native['all_objects_SHA256'].items())
 assert sha(pin['current_976f_library'])==pin['current_976f_library_SHA256']
 reuse=json.loads((S/'REUSED_OBJECTS.json').read_bytes());all_deps={};objects=[]
 for parent in reuse['source_packages']:
  root=Path(parent['root']);assert sha(root/'SOURCE_MANIFEST.json')==parent['manifest_SHA256'];mf=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
  assert mf['files_SHA256']==parent['files_SHA256']and all(sha(root/n)==h for n,h in parent['files_SHA256'].items())
 for item in reuse['objects']:
  assert sha(item['object'])==item['object_SHA256']and sha(item['MD_path'])==item['MD_SHA256']
  assert deps(Path(item['MD_path']).read_text())==set(item['deps_SHA256'])and all(sha(p)==h for p,h in item['deps_SHA256'].items())
  all_deps.update(item['deps_SHA256']);objects.append(item['object'])
 assert len(objects)==3 and sha(S/'router_math.hpp')==reuse['router_math_SHA256']
 q['reused_objects']=[{k:item[k]for k in ('object','object_SHA256','MD_path','MD_SHA256')}for item in reuse['objects']]
 q['reused_actual_MD_dependency_counts']=[len(item['deps_SHA256'])for item in reuse['objects']]
 argv=['g++','-std=c++17','-O3','-ffp-contract=off','-ffunction-sections','-fdata-sections','-pthread','-I'+str(S),'-MD','-MF',str(B/'cpu_math.d'),'-c',str(S/'cpu_math.cpp'),'-o',str(B/'cpu_math.o')]
 fresh=[x for x in argv if x!='-MD'];i=fresh.index('-MF');del fresh[i:i+2];i=fresh.index('-o');del fresh[i:i+2];fresh[fresh.index('-c')]='-M'
 before={p:sha(p)for p in deps(run(fresh,'pure-math-fresh-M',60))}
 for p,h in before.items():
  if Path(p).is_relative_to(S):continue
  if p in all_deps:assert h==all_deps[p],p
  else:assert not p.startswith('/refactor/'),('Unknown project dependency',p)
 run(argv,'pure-math-compile',100);assert deps((B/'cpu_math.d').read_text())==set(before)and all(sha(p)==h for p,h in before.items())
 exe=B/'CPU-pure-router-math';run(['g++',str(B/'cpu_math.o'),'-pthread','-Wl,--gc-sections','-o',str(exe)],'pure-math-link',30)
 dyn=run(['readelf','-d',str(exe)],'pure-math-dynamic',20);needed=[line.split('[',1)[1].split(']',1)[0]for line in dyn.splitlines()if'NEEDED'in line]
 allowed={'libstdc++.so.6','libm.so.6','libgcc_s.so.1','libc.so.6','ld-linux-x86-64.so.2'};assert set(needed)<=allowed,needed
 ld=run(['ldd',str(exe)],'pure-math-ldd',20);assert'not found'not in ld
 assert not any(x in ld.lower()for x in ('libinfini','libdeep','libtorch','flash_attn','libcuda','libhip','libgalaxy'))
 host_contract=json.loads(run([str(exe)],'pure-math-8thread-contract',30).splitlines()[-1]);assert host_contract['passed']and host_contract['actual_owner_threads']==8 and not host_contract['SDK_loaded']
 import semantic_reference
 semantics=semantic_reference.run(B/'Torch-CXX14',exe);assert semantics['passed']and semantics['actual_shared_CXX_math']
 # Actual owner/CABI driver source, with its explicitly fake device leaves only.
 parentS=Path(reuse['consumer_source_root']);sys.path.insert(0,str(parentS))
 consumer=json.loads(run(['python3','-B',str(parentS/'check_CPU.py')],'CPU-whole-consumer',45).splitlines()[-1]);assert consumer['passed']and consumer['nonzero_forward_not_counted_completed']
 link=pin['link_argv'][:];link[1]=objects[1];link.remove(pin['old_C4_object']);link.remove(pin['old_HIP_object']);link[1:1]=[objects[0],objects[2]]
 link=[pin['current_976f_library']if p=='/refactor/candidates/latest-family-native-r269/Deep/libDeep_latest_native_r269.so'else '-Wl,-rpath,'+str(Path(pin['current_976f_library']).parent)if p=='-Wl,-rpath,/refactor/candidates/latest-family-native-r269/Deep'else p for p in link]
 library=B/'libDeep_learned_router_suffix_r419.so';link[link.index('-o')+1]=str(library);link+=['-Wl,-Bsymbolic','-Wl,-soname,libDeep_learned_router_suffix_r419.so','-pthread'];run(link,'private-DSO-link-only',60)
 ld=run(['ldd','-r',str(library)],'private-DSO-ldd-r-NO-initializers',30);assert'not found'not in ld and'undefined symbol'not in ld
 nm=run(['nm','-D','--defined-only',str(library)],'private-DSO-nm',20);names=[x.split()[-1]for x in nm.splitlines()]
 required=['error','id_size','id','create','keys','load','finish','forward','project','same_logits','proof_info','proof_read','stats','dtype','destroy','shutdown'];assert all(names.count('deep_router_'+n+'_r419')==1 for n in required)
 assert'router_suffix_r419'in nm and not any(x in names for x in ('deep_owner_arm_r8900','deep_model_activation_arm_r8846','deep_moe_tail_set_r7211','deep_model_quant1_mode_r8886'))
 dyn=run(['readelf','-d',str(library)],'private-DSO-dynamic',20);DSOneeded=[line.split('[',1)[1].split(']',1)[0]for line in dyn.splitlines()if'NEEDED'in line];assert'libDeep_HC_T1_Model_r366.so'in DSOneeded
 assert all(sha(p)==v['SHA256']for p,v in pin['actual_DTK_bindings'].items())
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mp),new_HOST_compile_count=1,new_HIP_compile_count=0,reused_HOST_count=2,reused_HIP_count=1,production_objects_not_recompiled=True,pure_math_executable_SHA256=sha(exe),pure_math_DT_NEEDED=needed,pure_math_actual_MD_SHA256={p:sha(p)for p in before},actual_shared_CXX_and_Torch_CPU=semantics,actual_CPU_8thread_contract=host_contract,actual_CPU_whole_consumer=consumer,library=str(library),library_SHA256=sha(library),DT_NEEDED=DSOneeded,Bsymbolic=True,private_namespace=True,Native49objects_unchanged=True,current_Native_SHA256=sha(pin['current_976f_library']),GPU_GO=False,Model_GO=False,device_ABI_qualification=False,performance_trials=0)
except BaseException as exc:q.update(completed=True,passed=False,error=repr(exc),traceback=traceback.format_exc())
q['seconds']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ('passed','completed','actual_CPU','seconds','error')}));assert q['passed']
