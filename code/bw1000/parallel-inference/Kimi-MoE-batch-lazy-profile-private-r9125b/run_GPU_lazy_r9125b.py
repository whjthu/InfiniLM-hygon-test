import pathlib,json,hashlib,ctypes,time,contextlib,os,gc,sys
import subprocess
from Kimi_full_MoE_input_r8669s import load_fixture
from mapped_layer_weights_r9125 import mapped_layer_weights
import numpy as np
import torch
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/shared/Kimi-K3');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(manifest,output):
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
 m=json.loads(pathlib.Path(manifest).read_text());assert sha(m['library'])==m['library_SHA256'];lib=ctypes.CDLL(m['library']);lib.modern_kimi_TP16_moe_r8669w.argtypes=[ctypes.c_char_p];lib.modern_kimi_TP16_moe_result_r8669w.restype=ctypes.c_char_p
 out=pathlib.Path(output);assert not out.exists();raw=out.with_suffix('');raw.mkdir();d={'completed':False,'scope':'Actual651MoEL1 TP16 I192 privateT8..64 versus currentT16only; oneweightload; composite repeated16 historicalteacherrows; notModel/userthroughput','raw_directory':str(raw),'source_manifest_SHA256':sha(manifest),'phase':'loading_one_MoE_layer','node_rank':int(os.environ['SLURM_PROCID'])}
 def save():
  p=out.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(out)
 save()
 holder=None
 try:
  d['network_environment']={k:os.environ.get(k)for k in ['NCCL_NET_PLUGIN','NCCL_NET','NCCL_IB_DISABLE','NCCL_IB_HCA','NCCL_SOCKET_IFNAME','LD_LIBRARY_PATH','RDMAV_DRIVERS','UCX_MODULE_DIR','UCX_NET_DEVICES','UCX_TLS']};d['IB_device_visibility']={str(p):[v.name for v in p.iterdir()]if p.exists()else None for p in [pathlib.Path('/dev/infiniband'),pathlib.Path('/sys/class/infiniband')]};d['network_library_SHA256']={str(p):sha(p)for p in [pathlib.Path('/opt/dtk/lib/librccl.so.1'),pathlib.Path('/refactor/install/hygon-kimi-5f91ad7-r3672/lib/libinfiniccl.so'),*map(pathlib.Path,m['network_library_files'])]};save()
  class DlInfo(ctypes.Structure):_fields_=[('name',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
  dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];bindings=[]
  for path,expected,patterns in[(m['native'],m['native_SHA256'],['KimiK3MoE7forward','KimiK3MoEC1','CommunicationGroupC1','get_rank_infoEi','modern_kimi_I192_counts_r8669y','modern_kimi_grouped_I192_r8669y']),(m['Runtime'],m['Runtime_SHA256'],['Runtime14allocateMemory'])]:
   symbols=[line.split()[-1]for line in subprocess.check_output(['nm','-D','--defined-only',path],text=True).splitlines()if len(line.split())>=3]
   for pattern in patterns:
    names=[n for n in symbols if pattern in n];assert len(names)==1,(pattern,names);fn=getattr(lib,names[0]);info=DlInfo();assert dl.dladdr(ctypes.cast(fn,ctypes.c_void_p),ctypes.byref(info));actual=info.name.decode();assert sha(actual)==expected,(pattern,actual);bindings.append({'pattern':pattern,'DSO':actual,'SHA256':expected})
  d['actual_bindings']=bindings
  input_bytes,input_info=load_fixture(r,m['fixture'])
  config=json.loads((model/'config.json').read_text());mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];prefix='language_model.model.layers.1.block_sparse_moe.';groups={}
  for k,f in mapping.items():
   if k.startswith(prefix):groups.setdefault(f,[]).append(k)
  prior=json.loads(pathlib.Path(__file__).with_name('prior-L1-tensor-hash-bindings-r9125.json').read_text());assert sha(model/'config.json')==prior['config_SHA256']and sha(model/'model.safetensors.index.json')==prior['index_SHA256'];expected=json.loads(pathlib.Path(__file__).with_name('expected-headers-r9125.json').read_text());start=time.perf_counter();holder=mapped_layer_weights(model,prefix,expected);held,descriptors,proof=holder.__enter__();assert len(held)==5384 and {prefix+v['name']for v in descriptors}==set(prior['weight_SHA256']);d.update(CPU_weight_seconds=time.perf_counter()-start,CPU_weight_bytes=proof['logical_weight_bytes'],weight_count=len(held),mmap_source_proof=proof,config_SHA256=prior['config_SHA256'],index_SHA256=prior['index_SHA256'],historical_CPU_tensor_hash_producer=prior['producer'],current_full_weight_SHA_recomputed=False,CPU_weight_clones=0)
  input_bytes,input_info=load_fixture(r,m['fixture']);x=np.frombuffer(input_bytes,dtype=np.uint16).reshape(16,7168).copy();inputs={T:np.tile(x,((T+15)//16,1))[:T].copy()for T in [8,16,32,64]};d['input']=dict(input_info,construction={str(T):dict(rows=T,base_rows=16,distinct_binary_rows=len({row.tobytes()for row in v}),method='first8'if T==8 else'actual16'if T==16 else'repeated16x'+str(T//16),not_distinct_user_context=True)for T,v in inputs.items()})
  d['phase']='native_full_MoE_OFF_ON_warm_ABBA_BAAB';save();start=time.perf_counter();rc=lib.modern_kimi_TP16_moe_r8669w(json.dumps({'config':config,'weights':descriptors,'input_pointers':{str(T):int(v.ctypes.data)for T,v in inputs.items()},'raw_directory':str(raw),'node_rank':int(os.environ['SLURM_PROCID']),'master_addr':m['master_addr'],'master_port':m['master_port']}).encode());d['native_session_seconds']=time.perf_counter()-start;d['native']=json.loads(lib.modern_kimi_TP16_moe_result_r8669w());d['returncode']=rc;d['phase']='native_returned';save();assert rc==0 and d['native']['completed']
  d['raw_output_proof']=[{'file':p.name,'bytes':p.stat().st_size,'SHA256':sha(p)}for p in sorted(raw.glob('rank*-arm*.bin'))];assert len(d['raw_output_proof'])==320
  holder.__exit__(None,None,None);holder=None
  del held,descriptors,x,input_bytes;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();d['completed']=True;d['normal_CUDA_cleanup']=True;d['phase']='completed';save()
 except BaseException as e:d['error']=repr(e);save();raise
 finally:
  if holder is not None:holder.__exit__(None,None,None)
if __name__=='__main__':
 manifest=sys.argv[1];output='/refactor/results/Kimi-MoE-batch-lazy-profile-r9125b-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'-node'+os.environ['SLURM_PROCID']+'.json';run(manifest,output)