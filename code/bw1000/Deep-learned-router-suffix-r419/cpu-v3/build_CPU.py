"""Two private HOST TUs/one HIP object. CPU Torch and SDK metadata only; no Native execution."""
from pathlib import Path
import json,hashlib,subprocess,os,shlex,time,traceback
R=Path('/refactor');S=Path(__file__).resolve().parent;B=R/'candidates/Deep-learned-router-suffix-r419-cpu-v3';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,Runtime_created=False,weights_payload_opened=False,Model_loaded=False,SDK_linked_HOST_metadata_executable=True,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],publication_commit=os.environ['DEEP419_PUBLICATION_COMMIT'],commands=[])
start=time.monotonic()
def run(args,label,timeout=100):
 q['commands'].append(args);p=subprocess.run(args,capture_output=True,text=True,timeout=timeout);(B/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.stderr[-7000:],p.stdout[-1000:]);return p.stdout
def deps(text):return set(shlex.split(text.replace('\\\n',' ').split(':',1)[1]))
try:
 mp=S/'SOURCE_MANIFEST.json';assert sha(mp)==os.environ['DEEP419_CPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 pins=json.loads((S/'PARENT_PINS.json').read_bytes());assert sha(pins['current_976f_CPU_receipt'])==pins['current_976f_CPU_receipt_SHA256'];current=json.loads(Path(pins['current_976f_CPU_receipt']).read_bytes());assert current['passed']and len(current['all_objects_SHA256'])==49
 assert all(sha(p)==h for p,h in current['all_objects_SHA256'].items())and sha(pins['current_976f_library'])==pins['current_976f_library_SHA256']==current['library_SHA256']
 rp=Path(pins['original_MoE_CPU_receipt']);assert sha(rp)==pins['original_MoE_CPU_receipt_SHA256'];old=json.loads(rp.read_bytes());entry=next(x for x in old['new_compiled_units']if x['source'].endswith('deep_model_owner_r8900.cpp'))
 assert sha(entry['source'])==entry['source_SHA256']==pins['original_MoE_source_SHA256']and sha(entry['object'])==pins['original_MoE_object_SHA256']
 import rebase_source
 hp=Path(entry['source']).parent/'deepseek_v4_moe_routed_r20.hpp';assert sha(hp)==rebase_source.HEADER_SHA
 cpp,header,suffix=rebase_source.rebase(Path(entry['source']).read_text(),hp.read_text());assert cpp.encode()==(S/'private_moe.cpp').read_bytes()and header.encode()==(S/'private_moe.hpp').read_bytes()and suffix.encode()==(S/'ORIGINAL_SUFFIX.txt').read_bytes()
 history=entry['deps'];assert all(sha(p)==h for p,h in history.items())
 import semantic_reference
 pre=semantic_reference.run(B/'Torch-before-compile');assert pre['passed']and pre['Torch_CPU_reference']and not pre['GPU_initialized'];q['actual_Torch_CPU_before_compile']=pre
 objects=[];all_deps={}
 for name in ['private_moe','adapter']:
  argv=entry['compile_argv'][:];argv[1:1]=['-I'+str(S),'-I'+str(Path(entry['source']).parent),'-ffunction-sections','-fdata-sections','-ffp-contract=off'];argv[argv.index('-c')+1]=str(S/(name+'.cpp'));argv[argv.index('-o')+1]=str(B/(name+'.o'));argv[argv.index('-MF')+1]=str(B/(name+'.d'))
  fresh=[v for v in argv if v!='-MD'];i=fresh.index('-MF');del fresh[i:i+2];i=fresh.index('-o');del fresh[i:i+2];fresh[fresh.index('-c')]='-M'
  before={p:sha(p)for p in deps(run(fresh,name+'-fresh-M',60))}
  for p,h in before.items():
   if Path(p).is_relative_to(S):continue
   if p in history:assert h==history[p],p
   else:assert not p.startswith('/refactor/'),('Unknown project dependency',p)
  run(argv,name+'-compile',160);assert deps((B/(name+'.d')).read_text())==set(before)and all(sha(p)==h for p,h in before.items());objects.append(str(B/(name+'.o')));all_deps.update(before)
 hip=json.loads((S/'HIP_BUILD_PREFIX.json').read_bytes())['provider_prefix'][:];hip[1:1]=['-I'+str(S),'-ffp-contract=off'];hip+=['-MD','-MF',str(B/'router_kernel.d'),'-c',str(S/'router_kernel.cu'),'-o',str(B/'router_kernel.o')]
 fresh=[v for v in hip if v!='-MD'];i=fresh.index('-MF');del fresh[i:i+2];i=fresh.index('-o');del fresh[i:i+2];fresh[fresh.index('-c')]='-M';hd={p:sha(p)for p in deps(run(fresh,'router_kernel-fresh-M',60))}
 for p,h in hd.items():
  if Path(p).is_relative_to(S):continue
  if p in history:assert h==history[p],p
  else:assert not p.startswith('/refactor/'),('Unknown HIP project dependency',p)
 run(hip,'router_kernel-compile',120);assert deps((B/'router_kernel.d').read_text())==set(hd)and all(sha(p)==h for p,h in hd.items());all_deps.update(hd)
 # GC-link the real bridge main. It calls only compiled SDK config metadata and shared CPU math.
 sdk=R/'install/hygon-deep-5f91ad7-r3672/lib';exe=B/'CPU-router-math-and-metadata'
 run(['g++',objects[1],'-Wl,--gc-sections','-pthread','-L'+str(sdk),'-linfiniops','-linfinirt','-L/opt/dtk/cuda/cuda/lib64','-lcudart','-L/opt/dtk/lib','-lamdhip64','-Wl,-rpath,'+str(sdk),'-o',str(exe)],'CPU-executable-link',45)
 eld=run(['ldd',str(exe)],'CPU-executable-ldd',20);assert 'not found'not in eld and'libDeep_'not in eld and'libdeep_'not in eld and'libinfinilm_runtime'not in eld
 metadata=json.loads(run([str(exe)],'CPU-actual-shared-dispatch-and-SDK-metadata',30).splitlines()[-1]);assert metadata['passed']and metadata['GPU_calls']==0 and not metadata['Runtime_created']
 for op,c in metadata['SDK_default_metadata_only'].items():assert c['default']in c['active']and c['needs_resolution']is False,(op,c)
 post=semantic_reference.run(B/'Torch-after-compile',exe);assert post['passed']and post['actual_shared_CXX_math']
 consumers=json.loads(run(['python3','-B',str(S/'check_CPU.py')],'CPU-whole-consumer',45).splitlines()[-1]);assert consumers['passed']and consumers['nonzero_forward_not_counted_completed']
 link=pins['link_argv'][:];link[1]=objects[1];link.remove(pins['old_C4_object']);link.remove(pins['old_HIP_object']);link[1:1]=[objects[0],str(B/'router_kernel.o')]
 link=[pins['current_976f_library']if p=='/refactor/candidates/latest-family-native-r269/Deep/libDeep_latest_native_r269.so'else '-Wl,-rpath,'+str(Path(pins['current_976f_library']).parent)if p=='-Wl,-rpath,/refactor/candidates/latest-family-native-r269/Deep'else p for p in link]
 library=B/'libDeep_learned_router_suffix_r419.so';link[link.index('-o')+1]=str(library);link+=['-Wl,-Bsymbolic','-Wl,-soname,libDeep_learned_router_suffix_r419.so','-pthread'];run(link,'link',60)
 ld=run(['ldd','-r',str(library)],'ldd-r',30);assert 'not found'not in ld and'undefined symbol'not in ld
 nm=run(['nm','-D','--defined-only',str(library)],'nm',20);required=['error','id_size','id','create','keys','load','finish','forward','project','same_logits','proof_info','proof_read','stats','dtype','destroy','shutdown'];names=[x.split()[-1]for x in nm.splitlines()];assert all(names.count('deep_router_'+n+'_r419')==1 for n in required)
 assert 'router_suffix_r419'in nm and not any(name in names for name in ['deep_owner_arm_r8900','deep_model_activation_arm_r8846','deep_moe_tail_set_r7211','deep_model_quant1_mode_r8886'])
 dyn=run(['readelf','-d',str(library)],'dynamic',20);needed=[line.split('[',1)[1].split(']',1)[0]for line in dyn.splitlines()if'NEEDED'in line];assert any('libDeep_HC_T1_Model_r366.so'in v for v in needed)
 for p,v in pins['actual_DTK_bindings'].items():assert sha(p)==v['SHA256']
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mp),library=str(library),library_SHA256=sha(library),HOST_compile_count=2,HIP_compile_count=1,objects_SHA256={p:sha(p)for p in objects+[str(B/'router_kernel.o')]},dependencies_SHA256=all_deps,MD_equals_fresh_M=True,Native49objects_unchanged=True,current_Native_SHA256=sha(pins['current_976f_library']),SDK_bindings=pins['actual_DTK_bindings'],DT_NEEDED=needed,private_namespace=True,Bsymbolic=True,actual_shared_CXX_and_Torch_CPU=post,actual_SDK_default_metadata_and_dispatch=metadata,actual_CPU_whole_consumer=consumers,GPU_GO=False,Model_gain='UNKNOWN',performance_trials=0)
except BaseException as exc:q.update(completed=True,passed=False,error=repr(exc),traceback=traceback.format_exc())
q['HOST_compile_attempts']=sum(a[0]=='g++'and'-c'in a for a in q['commands']);q['HIP_compile_attempts']=sum(a[0]!='g++'and'-c'in a for a in q['commands']);q['seconds']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['passed','completed','actual_CPU','seconds','error']}));assert q['passed']
