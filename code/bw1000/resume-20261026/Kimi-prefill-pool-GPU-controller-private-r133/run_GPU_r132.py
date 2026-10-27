import pathlib,json,hashlib,ctypes,time,os,gc,sys,subprocess,contextlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from Kimi_full_MoE_input_r8669s import load_fixture
from mapped_layer_weights_r9125 import mapped_layer_weights
import numpy as np
import torch
r=pathlib.Path('/refactor');model=pathlib.Path('/shared/Kimi-K3');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(mp,output):
 assert os.environ.get('KIMI_PREFILL_POOL_COMPONENT_R132')=='1','component defaultOFF';torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0);assert len(os.sched_getaffinity(0))>=32
 m=json.loads(pathlib.Path(mp).read_bytes());assert m['component_enabled']is True;assert sha(m['library'])==m['library_SHA256']and sha(m['native'])==m['native_SHA256'];lib=ctypes.CDLL(m['library']);lib.modern_kimi_prefill_pool_run_r132.argtypes=[ctypes.c_char_p];lib.modern_kimi_prefill_pool_run_r132.restype=ctypes.c_int;lib.modern_kimi_prefill_pool_result_r132.restype=ctypes.c_char_p
 out=pathlib.Path(output);assert not out.exists();raw=out.with_suffix('');raw.mkdir();d={'completed':False,'phase':'mapping_singleL1','scope':'TP16 actualMoEL1 T576 longprefillshape, historical16teacher repeated36x, router recomputed; afterhelperinactivepool trimcandidate; NOT actual478activation/fullModel/PP orfitproof','node_rank':int(os.environ['SLURM_PROCID']),'manifest_SHA256':sha(mp)}
 def save():tmp=out.with_suffix('.tmp');tmp.write_bytes(json.dumps(d,indent=2).encode());tmp.replace(out)
 save()
 try:
  for p,h in m['source_SHA256'].items():assert sha(p)==h,p
  d['actual_network_library_SHA256']={p:sha(p)for p in m['network_library_files']};assert d['actual_network_library_SHA256']==m['network_library_files']
  class DI(ctypes.Structure):_fields_=[('name',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
  dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DI)];bindings=[]
  for symbol in ['modern_kimi_prefill_pool_trim_stats_r130','modern_kimi_W2_pair_counts_r9280','modern_kimi_grouped_I192_r8669y','modern_kimi_I192_T_counts_r9129']:
   info=DI();assert dl.dladdr(ctypes.cast(getattr(lib,symbol),ctypes.c_void_p),ctypes.byref(info));actual=info.name.decode();assert sha(actual)==m['native_SHA256'];bindings.append({'symbol':symbol,'actual_DSO':actual,'SHA256':m['native_SHA256']})
  info=DI();assert dl.dladdr(ctypes.cast(lib.modern_cpu_copy_stats_r2515,ctypes.c_void_p),ctypes.byref(info));actual=info.name.decode();assert sha(actual)==m['Runtime_SHA256'];info=DI();assert dl.dladdr(ctypes.cast(lib.modern_grouped_mxfp4_owned,ctypes.c_void_p),ctypes.byref(info));helper=pathlib.Path(info.name.decode());assert helper.resolve()==pathlib.Path(m['helper_expected_path']).resolve()and sha(helper)==m['helper_expected_SHA256'];d['actual_helper_binding']={'path':str(helper),'SHA256':sha(helper)};d['CPUcopy_interface_binding']={'actual_DSO':actual,'SHA256':sha(actual),'interface':'uint64[5] processglobal accumulated noreset'};d['actual_bindings']=bindings;d['phase']='loading_oneL1_lazymapped';save();expected=json.loads(pathlib.Path(__file__).with_name('expected-headers-r9125.json').read_bytes());tick=time.perf_counter()
  with mapped_layer_weights(model,'language_model.model.layers.1.block_sparse_moe.',expected)as(held,descriptors,proof):
   d['CPU_map_seconds']=time.perf_counter()-tick;d['CPU_weight_mapping']=proof;d['CPU_weight_bytes']=proof['logical_weight_bytes'];d['weight_count']=len(held);assert len(held)==5384;input_bytes,input_info=load_fixture(r,m['fixture']);x=np.frombuffer(input_bytes,np.uint16).reshape(16,7168).copy();inputs={T:np.tile(x,((T+15)//16,1))[:T].copy()for T in [576]};d['input']={'historical':input_info,'rows_source':'historical16 rows repeated, no claim currentrequests','construction':{str(T):{'rows':T,'distinct_binary_rows':len({row.tobytes()for row in a}),'repeated':T>16}for T,a in inputs.items()}};d['phase']='warmOFF2_then4ABBA_T576_pooltrim_pairs';save();tick=time.perf_counter();rc=lib.modern_kimi_prefill_pool_run_r132(json.dumps({'config':json.loads((model/'config.json').read_bytes()),'weights':descriptors,'input_pointers':{str(T):int(a.ctypes.data)for T,a in inputs.items()},'raw_directory':str(raw),'node_rank':int(os.environ['SLURM_PROCID']),'master_addr':m['master_addr'],'master_port':m['master_port']}).encode());d['native_session_s']=time.perf_counter()-tick;d['native']=json.loads(lib.modern_kimi_prefill_pool_result_r132());d['returncode']=rc;save();assert rc==0 and d['native']['completed']
   assert d['native']['all8normal_before_communicators']
   for owner in d['native']['reports']:
    assert owner['normal_Runtime_close']and owner['module_closed']and len(owner['owner_CPU_affinity'])==4 and len(owner['profile'])==0 and len(owner['arms'])==10 and len(owner['current_branch'])==1
    assert pathlib.Path(owner['actual_helper_path']).resolve()==pathlib.Path(m['helper_expected_path']).resolve()
    assert owner['installation_inclusive_wall_s']>0
    for sample in owner['arms']:assert sample['T']==576 and sample['finite']and sample['output_relative_RMS']<=.01 and sample['input_readonly_and_tail_canary']and sample['sampled_not_exact_peak']and sample['driver_samples']>0 and sample['native_graph_pinned_before']==sample['native_graph_pinned_after']==0
    assert [x['trim_after_helper']for x in owner['arms']]==[False,False,False,True,True,False,False,True,True,False]
    assert owner['current_branch'][0]['I192_calls']==owner['current_branch'][0]['W2pair_calls']==0
  del held,descriptors,x,input_bytes;gc.collect();torch.cuda.synchronize();d.update(completed=True,normal_CUDA_cleanup=True,phase='completed',Model_fit='PARK/UNKNOWN; component doesnotinclude93layerweights/PP/KDA/KV/head; sampledminnotexactpeak');save();print('KIMI_PREFILL_POOL_R132_NORMAL',d['node_rank'],flush=True)
 except BaseException as e:d['error']=repr(e);save();raise
if __name__=='__main__':run(sys.argv[1],'/refactor/results/Kimi-prefill-pool-r133-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'-node'+os.environ['SLURM_PROCID']+'.json')
