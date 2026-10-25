import pathlib,json,hashlib,ctypes,time,contextlib,os,gc,sys
import subprocess
from Kimi_full_MoE_input_r8669s import load_fixture
import numpy as np
import torch
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/shared/Kimi-K3');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(manifest,output):
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
 m=json.loads(pathlib.Path(manifest).read_text());assert sha(m['library'])==m['library_SHA256'];lib=ctypes.CDLL(m['library']);lib.modern_kimi_TP16_moe_r8669zx.argtypes=[ctypes.c_char_p];lib.modern_kimi_TP16_moe_result_r8669zx.restype=ctypes.c_char_p
 out=pathlib.Path(output);assert not out.exists();raw=out.with_suffix('');raw.mkdir();d={'completed':False,'scope':'Historicalteacherdecode6110/currentI192ON vs officialW13mode114 realTP16 pairedcomponent, notModelTPS','raw_directory':str(raw),'source_manifest_SHA256':sha(manifest),'phase':'loading_one_MoE_layer','node_rank':int(os.environ['SLURM_PROCID'])}
 def save():
  p=out.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(out)
 save()
 try:
  d['network_environment']={k:os.environ.get(k)for k in ['NCCL_NET_PLUGIN','NCCL_NET','NCCL_IB_DISABLE','NCCL_IB_HCA','NCCL_SOCKET_IFNAME','LD_LIBRARY_PATH','RDMAV_DRIVERS','UCX_MODULE_DIR','UCX_NET_DEVICES','UCX_TLS']};d['IB_device_visibility']={str(p):[v.name for v in p.iterdir()]if p.exists()else None for p in [pathlib.Path('/dev/infiniband'),pathlib.Path('/sys/class/infiniband')]};d['network_library_SHA256']={str(p):sha(p)for p in [pathlib.Path('/opt/dtk/lib/librccl.so.1'),pathlib.Path('/refactor/install/hygon-kimi-5f91ad7-r3672/lib/libinfiniccl.so'),*map(pathlib.Path,m['network_library_files'])]};save()
  class DlInfo(ctypes.Structure):_fields_=[('name',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
  dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];bindings=[]
  for path,expected,patterns in[(m['native'],m['native_SHA256'],['KimiK3MoE7forward','KimiK3MoEC1','CommunicationGroupC1','get_rank_infoEi','modern_kimi_I192_counts_r8669y','modern_kimi_grouped_I192_r8669y','modern_kimi_official_WFP4_r8669zu','modern_kimi_official_counts_r8669zu']),(m['Runtime'],m['Runtime_SHA256'],['Runtime14allocateMemory'])]:
   symbols=[line.split()[-1]for line in subprocess.check_output(['nm','-D','--defined-only',path],text=True).splitlines()if len(line.split())>=3]
   for pattern in patterns:
    names=[n for n in symbols if pattern in n];assert len(names)==1,(pattern,names);fn=getattr(lib,names[0]);info=DlInfo();assert dl.dladdr(ctypes.cast(fn,ctypes.c_void_p),ctypes.byref(info));actual=info.name.decode();assert sha(actual)==expected,(pattern,actual);bindings.append({'pattern':pattern,'DSO':actual,'SHA256':expected})
  d['actual_bindings']=bindings
  input_bytes,input_info=load_fixture(r,m['fixture'])
  config=json.loads((model/'config.json').read_text());mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];prefix='language_model.model.layers.1.block_sparse_moe.';groups={}
  for k,f in mapping.items():
   if k.startswith(prefix):groups.setdefault(f,[]).append(k)
  held=[];descriptors=[];weight_hashes={};start=time.perf_counter()
  for fn,keys in sorted(groups.items()):
   with safe_open(model/fn,framework='pt',device='cpu')as f:
    for key in sorted(keys):
     v=f.get_tensor(key).clone().contiguous();assert v.dtype in[torch.float32,torch.bfloat16,torch.uint8];held.append(v);descriptors.append({'name':key[len(prefix):],'pointer':v.data_ptr(),'shape':list(v.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[v.dtype]});weight_hashes[key]=hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()
  d['CPU_weight_seconds']=time.perf_counter()-start;d['CPU_weight_bytes']=sum(v.numel()*v.element_size()for v in held);d['weight_count']=len(held);d['weight_SHA256']=weight_hashes;d['config_SHA256']=sha(model/'config.json');d['index_SHA256']=sha(model/'model.safetensors.index.json')
  input_bytes,input_info=load_fixture(r,m['fixture']);x=np.frombuffer(input_bytes,dtype=np.uint16).reshape(16,7168).copy();d['input']=input_info
  d['phase']='native_full_MoE_OFF_ON_warm_ABBA_BAAB';save();start=time.perf_counter();rc=lib.modern_kimi_TP16_moe_r8669zx(json.dumps({'config':config,'weights':descriptors,'input_pointer':int(x.ctypes.data),'raw_directory':str(raw),'node_rank':int(os.environ['SLURM_PROCID']),'master_addr':m['master_addr'],'master_port':m['master_port']}).encode());d['native_session_seconds']=time.perf_counter()-start;d['native']=json.loads(lib.modern_kimi_TP16_moe_result_r8669zx());d['returncode']=rc;d['phase']='native_returned';save();assert rc==0 and d['native']['completed']
  d['raw_output_proof']=[{'file':p.name,'bytes':p.stat().st_size,'SHA256':sha(p)}for p in sorted(raw.glob('rank*-arm*.bin'))];assert len(d['raw_output_proof'])==80
  del held,descriptors,x,input_bytes;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();d['completed']=True;d['normal_CUDA_cleanup']=True;d['phase']='completed';save()
 except BaseException as e:d['error']=repr(e);save();raise
if __name__=='__main__':
 manifest=sys.argv[1];output='/refactor/results/Kimi-official-WFP4-r8669zy-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'-node'+os.environ['SLURM_PROCID']+'.json';run(manifest,output)