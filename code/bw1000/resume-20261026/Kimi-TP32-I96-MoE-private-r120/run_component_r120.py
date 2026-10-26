"""Future root-authorized standalone component consumer; no Slurm launch code."""
import pathlib,json,hashlib,ctypes,time,os,sys,gc,subprocess
from mapped_layer_weights_r9125 import mapped_layer_weights
from Kimi_full_MoE_input_r8669s import load_fixture
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(manifest_path,output):
 import numpy as np
 import torch
 m=json.loads(pathlib.Path(manifest_path).read_bytes());D=pathlib.Path(__file__).resolve().parent;source=json.loads((D/'source-manifest-r120.json').read_bytes())
 assert m['source_manifest_SHA256']==sha(D/'source-manifest-r120.json') and m['explicit_component_GO'] is True
 assert m['nodes']==['a02r4n08','a02r4n09','a02r4n10','a02r4n11'] and m['target_cards']==32 and m['parent_allocation']=='910375'
 assert m['TP']==32 and m['PP']==1 and m['layer']==1 and m['T']==[8,16,64]
 node=int(os.environ['SLURM_PROCID']);assert 0<=node<4 and os.uname().nodename==m['nodes'][node]
 assert m['master_addr']!='FILL_FROM_FRESH_IB0' and 1024<int(m['master_port'])<65535
 assert len(os.sched_getaffinity(0))>=32
 torch.set_num_threads(1)
 for p,h in source['pinned_libraries'].items():assert sha(p)==h,p
 for n,h in source['package_source_SHA256'].items():assert sha(D/n)==h,n
 cpu=json.loads(pathlib.Path(m['CPU_receipt']).read_bytes());assert cpu['completed'] and cpu['CPU_only'] and not cpu['GPU'] and cpu['native_SHA256']==source['native_SHA256']
 assert sha(m['CPU_receipt'])==m['CPU_receipt_SHA256'] and sha(cpu['library'])==cpu['library_SHA256']==m['library_SHA256']
 # The explicit future GO and existing audit/keeper must precede this CUDA init.
 assert torch.cuda.device_count()==8
 torch.cuda.set_device(0)
 def memory():return [list(torch.cuda.mem_get_info(i)) for i in range(8)]
 out=pathlib.Path(output);assert not out.exists();raw=out.with_suffix('');raw.mkdir()
 report={'completed':False,'node_rank':node,'scope':source['scope'],'phase':'preflight','source_manifest_SHA256':sha(D/'source-manifest-r120.json'),'manifest_SHA256':sha(manifest_path),'memory_before':memory()}
 def save():p=out.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(out)
 def floor(v):assert min(x[0] for x in v)>4*1024**3,'unchanged4GiB floor'
 floor(report['memory_before']);save()
 try:
  lib=ctypes.CDLL(cpu['library']);lib.modern_kimi_TP32_moe_r120.argtypes=[ctypes.c_char_p];lib.modern_kimi_TP32_moe_r120.restype=ctypes.c_int;lib.modern_kimi_TP32_moe_result_r120.restype=ctypes.c_char_p
  class DI(ctypes.Structure):_fields_=[('path',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('addr',ctypes.c_void_p)]
  dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DI)];dl.dladdr.restype=ctypes.c_int;bindings=[]
  for path,expected,patterns in [(source['native'],source['native_SHA256'],['KimiK3MoE7forward','KimiK3MoEC1','CommunicationGroupC1','get_rank_infoEi','modern_kimi_I192_counts_r8669y','modern_kimi_set_grouped_prefill','modern_kimi_W2_pair_counts_r9280']),(source['Runtime'],source['Runtime_SHA256'],['Runtime14allocateMemory'])]:
   names=[l.split()[-1] for l in subprocess.check_output(['nm','-D','--defined-only',path],text=True).splitlines() if len(l.split())>=3]
   for pattern in patterns:
    candidates=[n for n in names if pattern in n];assert len(candidates)==1,(pattern,candidates);v=DI();assert dl.dladdr(ctypes.cast(getattr(lib,candidates[0]),ctypes.c_void_p),ctypes.byref(v));actual=v.path.decode();assert sha(actual)==expected,(pattern,actual);bindings.append({'symbol':candidates[0],'actual_DSO':actual,'SHA256':expected})
  report['actual_bindings']=bindings
  actual_maps={}
  for pinned,expected in source['pinned_libraries'].items():
   name=pathlib.Path(pinned).name
   if name not in ['libinfiniops.so','libinfinirt.so','libinfiniccl.so','libinfinilm_runtime.so']:continue
   hits={line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if pathlib.Path(line.split()[-1]).name==name};assert len(hits)==1,(name,hits);actual=next(iter(hits));assert sha(actual)==expected,(name,actual);actual_maps[name]={'path':actual,'SHA256':expected}
  assert set(actual_maps)=={'libinfiniops.so','libinfinirt.so','libinfiniccl.so','libinfinilm_runtime.so'};report['actual_loaded_core_libraries']=actual_maps
  model=pathlib.Path('/shared/Kimi-K3');assert sha(model/'config.json')==source['config_SHA256'] and sha(model/'model.safetensors.index.json')==source['index_SHA256']
  data,fixture=load_fixture('/refactor',source['fixture']);x=np.frombuffer(data,np.uint16).reshape(16,7168).copy();inputs={T:np.tile(x,((T+15)//16,1))[:T].copy() for T in [8,16,64]}
  report['input']={'historical':fixture,'base_binary_distinct_rows':len({row.tobytes() for row in x}),'construction':{str(T):{'rows':T,'distinct_binary_rows':len({r.tobytes() for r in v}),'method':'first8' if T==8 else 'original16' if T==16 else 'repeat_original16_four_times','not_current_real_users':True} for T,v in inputs.items()}}
  expected=json.loads((D/'expected-headers-r9125.json').read_bytes());report['phase']='mapping_one_layer_no_clone';save()
  with mapped_layer_weights(model,'language_model.model.layers.1.block_sparse_moe.',expected) as (held,desc,proof):
   assert len(held)==5384;report['CPU_weight_mapping']=proof;report['phase']='native_single_layer_load_and_repeat';save();tick=time.monotonic()
   payload={'config':json.loads((model/'config.json').read_bytes()),'weights':desc,'input_pointers':{str(T):int(v.ctypes.data) for T,v in inputs.items()},'raw_directory':str(raw),'node_rank':node,'master_addr':m['master_addr'],'master_port':m['master_port']}
   rc=lib.modern_kimi_TP32_moe_r120(json.dumps(payload).encode());report['native_session_s']=time.monotonic()-tick;report['native']=json.loads(lib.modern_kimi_TP32_moe_result_r120());report['returncode']=rc;save();assert rc==0 and report['native']['completed'] and report['native']['all8normal_before_communicators']
   assert [o['global_tp_rank'] for o in report['native']['reports']]==list(range(node*8,(node+1)*8))
   for o in report['native']['reports']:assert o['module_closed'] and o['normal_Runtime_close'] and len(o['arms'])==18 and all(a['nonfinite']==0 and a['input_readonly_tail_canary'] and a['private_provider_counter_delta_zero'] for a in o['arms'])
  files=sorted(raw.glob('globalrank*-arm*.bin'));assert len(files)==8*3*6
  report['raw_output_proof']=[{'file':p.name,'bytes':p.stat().st_size,'SHA256':sha(p)} for p in files]
  del held,desc,x,data;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();report['memory_after_normal_close']=memory();floor(report['memory_after_normal_close']);report.update(completed=True,normal_CUDA_cleanup=True,phase='completed');save()
 except BaseException as e:report['error']=repr(e);save();raise
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
