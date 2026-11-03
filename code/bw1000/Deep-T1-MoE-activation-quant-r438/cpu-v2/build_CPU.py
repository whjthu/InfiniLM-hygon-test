"""Private 3 HOST + 1 HIP production build; CPU executes standard-library math only."""
from pathlib import Path
import hashlib,json,os,re,shlex,subprocess,time,traceback
S=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/Deep-T1-MoE-activation-quant-r438-cpu-v2'
B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,CPU_only=True,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],publication_commit=os.environ['DEEP438_PUBLICATION_COMMIT'],commands=[],Native_dlopen=False,Runtime_created=False,GPU_initialized=False,weights_payload_opened=False,actual_SDK_metadata_executed=False,effective_LD_PRELOAD=os.environ.get('LD_PRELOAD'),GPU_GO=False,Model_GO=False)
start=time.monotonic()
def save():
 q['seconds']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n')
def run(args,label,timeout=100):
 q['active_phase']=label;q['commands'].append(args);save()
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 (B/(label+'.log')).write_text(p.stdout+p.stderr)
 assert p.returncode==0,(label,p.stderr[-7000:],p.stdout[-1500:]);return p.stdout
def deps(text):return set(shlex.split(text.replace(chr(92)+chr(10),' ').split(':',1)[1]))
def make_fresh(argv):
 a=[x for x in argv if x!='-MD'];i=a.index('-MF');del a[i:i+2];i=a.index('-o');del a[i:i+2];a[a.index('-c')]='-M';return a
try:
 mp=S/'SOURCE_MANIFEST.json';assert sha(mp)==os.environ['DEEP438_CPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 assert not q['effective_LD_PRELOAD'],'No environment override of SDK initialization is allowed'
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 pin=json.loads((S/'PARENT_PINS.json').read_bytes());assert sha(pin['Native_build'])==pin['Native_build_SHA256']
 family=json.loads(Path(pin['Native_build']).read_bytes())['families']['Deep'];native=family['Native'];worker=family['Native_worker']
 assert sha(pin['Native'])==pin['Native_SHA256'] and len(native['unchanged_objects_SHA256'])==47
 assert all(sha(p)==h for p,h in native['unchanged_objects_SHA256'].items())
 # Preserve the exact successful native worker/object graph, not 976f's additional object.
 assert worker['new_object_SHA256']==sha(worker['new_object'])
 q['Native_unchanged_objects_SHA256']=native['unchanged_objects_SHA256'];q['Native_worker']={k:worker[k]for k in ['new_object','new_object_SHA256','new_source','new_source_SHA256']};q['Native_object_count']=48
 parent=[];history={}
 for item in pin['parent_CPU_receipts']:
  assert sha(item['path'])==item['SHA256'];parent.append(json.loads(Path(item['path']).read_bytes()))
 moe=next(x for x in parent[0]['new_compiled_units']if x['source'].endswith('deep_model_owner_r8900.cpp'))
 bank=next(x for x in parent[1]['compiled_units']if x['source'].endswith('deepseek_v4_int8_mlp_bank_r20.cpp'))
 for entry in [moe,bank]:
  assert sha(entry['source'])==entry['source_SHA256'] and sha(entry['object'])==entry['object_SHA256']
  for p,h in entry['deps'].items():
   assert sha(p)==h
   if p in history:assert history[p]==h
   history[p]=h
 for p,h in json.loads((S/'JSON_HEADER_PINS.json').read_bytes()).items():assert sha(p)==h;history[p]=h
 for p,v in pin['actual_DTK_bindings'].items():assert sha(p)==v['SHA256']
 q['original_parent_MD_dependency_counts']=[len(moe['deps']),len(bank['deps'])]
 reuse=json.loads((S/'REUSED_OBJECTS.json').read_bytes());oldS=Path(reuse['original_source_root'])
 assert sha(oldS/'SOURCE_MANIFEST.json')==reuse['original_source_manifest_SHA256']
 oldmf=json.loads((oldS/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(oldS/n)==h for n,h in oldmf['files_SHA256'].items())
 assert sha(reuse['original_CPU_receipt'])==reuse['original_CPU_receipt_SHA256'];prior=json.loads(Path(reuse['original_CPU_receipt']).read_bytes())
 assert prior['completed']and not prior['passed']and prior['actual_CPU']=='910407.888'and prior['active_phase']=='adapter-compile'
 assert prior['actual_Torch_CXX_CPU']['passed']and len(prior['actual_Torch_CXX_CPU']['cases'])==10 and all(x['passed']for x in prior['actual_Torch_CXX_CPU']['cases'])
 assert prior['actual_typed_driver_recording_consumer']['passed']
 guard=reuse['device_guard'];assert sha(guard['path'])==guard['SHA256']=='06b1ad7ec4920e906c0bace4b17b67e579139cd016bf7d50e16e05d5f3ba5712';history[guard['path']]=guard['SHA256']
 q['CPU888_preserved_failed']=True;q['actual_reused_CPU_semantics']=prior['actual_Torch_CXX_CPU'];q['actual_reused_typed_consumer']=prior['actual_typed_driver_recording_consumer']
 # CPU contract/independent Torch must pass before any production compile.
 objects=[];all_deps={};compiled=[]
 def compile_one(argv,name):
  before={p:sha(p)for p in deps(run(make_fresh(argv),name+'-fresh-M',65))}
  for p,h in before.items():
   if Path(p).is_relative_to(S):continue
   if p in history:assert h==history[p],p
   else:assert not p.startswith('/refactor/'),('Unknown project dependency',p)
  run(argv,name+'-compile',160)
  md=Path(argv[argv.index('-MF')+1]);assert deps(md.read_text())==set(before)and all(sha(p)==h for p,h in before.items())
  obj=argv[argv.index('-o')+1];compiled.append(dict(name=name,object=obj,object_SHA256=sha(obj),MD=str(md),MD_SHA256=sha(md),deps_SHA256=before,compile_argv=argv,fresh_M_equals_MD=True));all_deps.update(before);return obj
 # No successful CPU888 compile, semantic case or consumer is replayed.
 for item in reuse['objects']:
  assert sha(item['object'])==item['object_SHA256']and sha(item['MD'])==item['MD_SHA256']
  names=deps(Path(item['MD']).read_text());assert len(names)==item['dependency_count']
  for p in names:
   if p in item['project_deps_SHA256']:assert sha(p)==item['project_deps_SHA256'][p]
   elif p in history:assert sha(p)==history[p]
   else:assert not p.startswith('/refactor/'),('Unknown reused project dependency',p)
  objects.append(item['object'])
 assert len(objects)==2;exe=Path(reuse['pure_math_executable']['path']);assert sha(exe)==reuse['pure_math_executable']['SHA256']
 q['reused_production_HOST_objects']=reuse['objects'];q['pure_math_reused_not_executed']=reuse['pure_math_executable']
 for name,entry in [('adapter',moe)]:
  argv=entry['compile_argv'][:]
  argv[1:1]=['-I'+str(S),'-I'+str(Path(entry['source']).parent),'-ffp-contract=off']
  argv[argv.index('-c')+1]=str(S/(name+'.cpp'));argv[argv.index('-o')+1]=str(B/(name+'.o'));argv[argv.index('-MF')+1]=str(B/(name+'.d'))
  objects.append(compile_one(argv,name))
 hip=json.loads((S/'HIP_BUILD_PREFIX.json').read_bytes())['provider_prefix'][:]
 hip[1:1]=['-I'+str(S),'-ffp-contract=off'];hip+=['-MD','-MF',str(B/'activation_quant.d'),'-c',str(S/'activation_quant.cu'),'-o',str(B/'activation_quant.o')]
 objects.append(compile_one(hip,'activation_quant'))
 link=pin['link_argv'][:];link[1:4]=objects;library=B/'libDeep_T1_MoE_activation_quant_r438.so';link[link.index('-o')+1]=str(library)
 link+=['-Wl,-Bsymbolic','-Wl,-soname,libDeep_T1_MoE_activation_quant_r438.so','-pthread'];run(link,'private-DSO-link-NO-execution',60)
 ld=run(['ldd','-r',str(library)],'private-DSO-ldd-r-NO-initializers',30);assert'not found'not in ld and'undefined symbol'not in ld
 resolved={name:path for name,path in re.findall(r'^\s*(\S+)\s+=>\s+(\S+)\s+\(',ld,re.M)}
 q['resolved_pinned_providers']={}
 for path,item in pin['actual_DTK_bindings'].items():
  base=Path(path).name;actual=resolved.get(base);assert actual,('Pinned dependency not in actual ldd closure',base)
  assert sha(actual)==item['SHA256'],(base,actual)
  q['resolved_pinned_providers'][base]=dict(path=actual,SHA256=sha(actual))
 nm=run(['nm','-D','--defined-only',str(library)],'private-DSO-nm',20);names=[x.split()[-1]for x in nm.splitlines()]
 from interface import SPECS
 assert all(names.count('deep_actquant_'+n+'_r438')==1 for n in SPECS)
 assert'actquant_r438'in nm and not any(x in names for x in ['deep_owner_arm_r8900','deep_model_activation_arm_r8846','deep_moe_tail_set_r7211','deep_model_quant1_mode_r8886'])
 dyn=run(['readelf','-d',str(library)],'private-DSO-dynamic',20);DSOneeded=[x.split('[',1)[1].split(']',1)[0]for x in dyn.splitlines()if'NEEDED'in x]
 native_dyn=run(['readelf','-d',pin['Native']],'current-Native-dynamic',20)
 sonames=[x.split('[',1)[1].split(']',1)[0]for x in native_dyn.splitlines()if'SONAME'in x];assert len(sonames)<=1
 exact_native_needed=sonames[0]if sonames else pin['Native'];assert exact_native_needed in DSOneeded,(exact_native_needed,DSOneeded)
 assert sha(pin['Native'])==pin['Native_SHA256'] and all(sha(p)==v['SHA256']for p,v in pin['actual_DTK_bindings'].items())
 q.update(passed=True,library=str(library),library_SHA256=sha(library),compiled=compiled,production_HOST_compile_count=1,production_HIP_compile_count=1,pure_HOST_compile_count=0,reused_production_HOST_count=2,pure_math_executable_SHA256=sha(exe),DT_NEEDED=DSOneeded,pinned_Native_DT_NEEDED=exact_native_needed,Native48objects_unchanged=True,private_namespace=True,Bsymbolic=True,actual_reused_CPU_8thread_contract=prior['actual_typed_driver_recording_consumer'],source_manifest_SHA256=sha(mp),SDKmetadata_deferred_to_same_future_device_component=True,device_ABI_qualification=False,performance_trials=0)
except BaseException as exc:q.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
q['completed']=True;q['HOST_compile_attempts']=sum(a[0]=='g++'and'-c'in a for a in q['commands']);q['HIP_compile_attempts']=sum(a[0]!='g++'and'-c'in a for a in q['commands']);save();print(json.dumps({k:q.get(k)for k in ['passed','completed','actual_CPU','seconds','active_phase','error']}));assert q['passed']
