"""Controller-bound wrapper around unchanged no-weight constructor driver."""
import pathlib,json,os,sys,subprocess
from contract_r120k import gate,sha,COMMON,COMMON_SHA
m,g,c,w=gate(os.environ['R120K_MANIFEST_SHA256'],os.environ['R120K_GO_FILE'],container=True)
runtime=pathlib.Path(os.environ['R120K_RUNTIME']);assert sha(runtime)==os.environ['R120K_RUNTIME_SHA256'];q=json.loads(runtime.read_bytes())
assert q['explicit_constructor_GO']is True and q['root_source_published']is True and q['publication_commit']==g['publication_commit']and q['controller_manifest_SHA256']==os.environ['R120K_MANIFEST_SHA256']
assert q['Native_CPU_receipt_SHA256']==g['Native_CPU_receipt_SHA256']and q['wrapper_CPU_receipt_SHA256']==g['wrapper_CPU_receipt_SHA256']and q['wrapper_SHA256']==g['wrapper_SHA256']
assert q['fresh_owned32_before']['rows'][0]['ib0_ipv4'][0]==q['master_addr']and q['root_GO_SHA256']==sha(os.environ['R120K_GO_FILE'])
# Resolve the actual constructor/Runtime providers before invoking any Runtime.
# These imports/dlopen are reached only inside the root-authorized GPU step.
import torch,ctypes
assert torch.cuda.device_count()==8
lib=ctypes.CDLL(w['library'])
class DI(ctypes.Structure):_fields_=[('path',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('addr',ctypes.c_void_p)]
dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DI)];dl.dladdr.restype=ctypes.c_int;bindings=[]
for path,expected,patterns in [(w['library'],w['library_SHA256'],['modern_kimi_TP32_construct_r120f']),(c['native'],c['native_SHA256'],['KimiK3DeltaAttentionC1','KimiK3MLAAttentionC1','kimi_k3_allocate_cache_tensors','CommunicationGroupC1'])]:
 names=[line.split()[-1]for line in subprocess.check_output(['nm','-D','--defined-only',path],text=True).splitlines()if len(line.split())>=3]
 for pattern in patterns:
  matches=[n for n in names if pattern in n];assert len(matches)==1,(pattern,matches);v=DI();assert dl.dladdr(ctypes.cast(getattr(lib,matches[0]),ctypes.c_void_p),ctypes.byref(v));actual=v.path.decode();assert sha(actual)==expected,(pattern,actual);bindings.append({'symbol':matches[0],'actual_DSO':actual,'SHA256':expected})
for item in w['actual_Common_provider_audit']:
 name=item['undefined'];v=DI();assert dl.dladdr(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(v));actual=v.path.decode();assert sha(actual)==COMMON_SHA,(item['demangled'],actual);bindings.append({'symbol':name,'demangled':item['demangled'],'actual_DSO':actual,'SHA256':COMMON_SHA})
loaded={}
for name in ['libinfiniops.so','libinfinirt.so','libinfiniccl.so','libinfinilm_runtime.so',pathlib.Path(COMMON).name]:
 expected=COMMON_SHA if name==pathlib.Path(COMMON).name else next(h for p,h in c['required_DSO_SHA256'].items()if pathlib.Path(p).name==name);hits={line.split()[-1]for line in pathlib.Path('/proc/self/maps').read_text().splitlines()if pathlib.Path(line.split()[-1]).name==name};assert len(hits)==1,(name,hits);actual=next(iter(hits));assert sha(actual)==expected;loaded[name]={'path':actual,'SHA256':expected}
maps=pathlib.Path('/proc/self/maps').read_text().splitlines();mapped_names={pathlib.Path(line.split()[-1]).name for line in maps};assert pathlib.Path(c['native']).name in mapped_names and 'libKimi_W2_pair_r9280.so'not in mapped_names and 'libKimi_TP32_construct_self_r120h.so'not in mapped_names,'unexpected second private Native/selfcontained route'
node=int(os.environ['SLURM_PROCID']);assert os.environ['SLURM_JOB_ID']=='910375'and 0<=node<4 and os.uname().nodename==m['nodes'][node]and len(os.sched_getaffinity(0))>=32
fresh=q['fresh_owned32_before'];assert fresh['qualified']and fresh['KFD_empty_all4']and fresh['port_free_all4']and fresh['nodes']==m['nodes']
import datetime
assert 0<=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(fresh['UTC'])).total_seconds()<=120
cfg=pathlib.Path('/shared/Kimi-K3/config.json');assert sha(cfg)==q['config_SHA256']=='9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213'
lib.modern_kimi_TP32_construct_r120f.argtypes=[ctypes.c_char_p];lib.modern_kimi_TP32_construct_r120f.restype=ctypes.c_int;lib.modern_kimi_TP32_construct_result_r120f.restype=ctypes.c_char_p
rc=lib.modern_kimi_TP32_construct_r120f(json.dumps({'node_rank':node,'master_addr':q['master_addr'],'master_port':q['master_port'],'config':json.loads(cfg.read_bytes())}).encode());raw=lib.modern_kimi_TP32_construct_result_r120f();assert raw;a=json.loads(raw)
a.update(node_rank=node,returncode=rc,CPU_receipt_SHA256=g['Native_CPU_receipt_SHA256'],Native_SHA256=c['native_SHA256'],Common_SHA256=COMMON_SHA,wrapper_CPU_receipt_SHA256=g['wrapper_CPU_receipt_SHA256'],wrapper_SHA256=w['library_SHA256'],actual_wrapper_provider_SHA256=w['library_SHA256'],actual_private_provider_SHA256=c['native_SHA256'],actual_Common_provider_SHA256=COMMON_SHA,provider_mode=w['provider_mode'],private_Native0f_loaded=True,rejected_selfcontained_DSO_loaded=False,runtime_manifest_SHA256=sha(runtime),source_manifest_SHA256=c['source_manifest_SHA256'],publication_commit=g['publication_commit'],controller_manifest_SHA256=os.environ['R120K_MANIFEST_SHA256'],actual_constructor_bindings=bindings,actual_loaded_core_libraries=loaded,memory_observation_scope='four CUDA phase samples per owner, not continuous/peak',graphPinned='UNKNOWN_NOT_SAMPLED',whole_run_memory_peak='UNKNOWN',fullModel_budget='UNKNOWN_STOP',Model_promotion=False)
p=pathlib.Path('/refactor/results')/('Kimi-TP32-construct-r120k-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'-node'+str(node)+'.json')
with p.open('x')as f:json.dump(a,f,indent=2)
assert rc==0 and a['completed']and a['all8normal_before_communicators']and not a['full93_model_parameters_constructed']and a['forward_calls']==0
print('KIMI_TP32_CONSTRUCT_R120K_NORMAL',flush=True)
