"""One real32rank single-layer component; no Slurm submission or full Model."""
from pathlib import Path
import ctypes,hashlib,json,os,socket,time,gc,sys
import numpy as np
import torch
from mapped_layer_weights_r9125 import mapped_layer_weights
from Kimi_full_MoE_input_r8669s import load_fixture
D=Path(__file__).resolve().parent;R=Path('/refactor');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(path):
 m=json.loads(Path(path).read_bytes());assert m['GPU_authorized']is True and m['source_published']is True and m['publication_branch']=='bw1000-experiments'
 assert m['parent_job']=='910380'and m['expected_nodes']==['a14r2n08','a14r2n09','a14r2n10','a14r2n12']
 assert os.environ['SLURM_JOB_ID']==m['parent_job']and int(os.environ['SLURM_NTASKS'])==4
 node=int(os.environ['SLURM_PROCID']);assert 0<=node<4 and socket.gethostname()==m['expected_nodes'][node]
 assert len(os.sched_getaffinity(0))>=32
 assert sha(D/'source-manifest-r163.json')==m['source_manifest_SHA256']
 for n,h in json.loads((D/'source-manifest-r163.json').read_bytes())['files_SHA256'].items():assert sha(D/n)==h,n
 bp=Path(m['CPU_build_path']);assert sha(bp)==m['CPU_build_SHA256'];b=json.loads(bp.read_bytes());assert b['completed']and b['CPU_only']and not b['GPU']and not b['Native_loaded']
 assert b['source_manifest_SHA256']==sha(D/'source-manifest-r163.json')and sha(b['native'])==b['native_SHA256']and sha(b['library'])==b['library_SHA256']
 ctx=json.loads((D/'parent-GPU-context-r155.json').read_bytes())
 for p,h in ctx['pinned_libraries'].items():assert sha(p)==h,p
 for p,h in ctx['network_library_files'].items():assert sha(p)==h,p
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
 def memory():return [list(torch.cuda.mem_get_info(i))for i in range(8)]
 def floor(v):assert min(x[0]for x in v)>4*1024**3,'unchanged4GiB floor'
 out=R/'results'/('Kimi-TP32-I96-template-r163-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'-node%d.json'%node);assert not out.exists()
 report={'completed':False,'node':node,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'scope':'Layer1 actualTP32 genericI96 vs TT2Tile4I96/W2pair; nofullModel/current449routes','source_manifest_SHA256':sha(D/'source-manifest-r163.json'),'Native_SHA256':b['native_SHA256'],'memory_before':memory()};floor(report['memory_before'])
 def save():tmp=out.with_suffix('.tmp');tmp.write_bytes((json.dumps(report,indent=2)+'\n').encode());tmp.replace(out)
 save()
 lib=ctypes.CDLL(b['library']);lib.modern_kimi_TP32_moe_r163.argtypes=[ctypes.c_char_p];lib.modern_kimi_TP32_moe_r163.restype=ctypes.c_int;lib.modern_kimi_TP32_moe_result_r163.restype=ctypes.c_char_p
 class DI(ctypes.Structure):_fields_=[('path',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('addr',ctypes.c_void_p)]
 dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DI)];proof=[]
 for name in ['modern_kimi_I96_mode_r163','modern_kimi_I96_counts_r163','modern_kimi_grouped_I96_r163']:
  z=DI();assert dl.dladdr(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(z));assert sha(z.path.decode())==b['native_SHA256'];proof.append({'symbol':name,'actual_DSO_SHA256':sha(z.path.decode())})
 report['actual_bindings']=proof;save()
 raw,fixture=load_fixture(R,ctx['fixture']);x=np.frombuffer(raw,np.uint16).reshape(16,7168).copy();inputs={str(T):np.tile(x,((T+15)//16,1))[:T].copy()for T in [8,16,64]}
 values=(x.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(values).all();rms=float(np.sqrt(np.mean(values.astype(np.float64)**2)));assert rms>0
 noise=(np.random.default_rng(16320261028).standard_normal((64,7168))*rms).astype(np.float32);u=noise.view(np.uint32);inputs['Gaussian64']=((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)
 report['input']={'historical':fixture,'historical_distinct_rows':len({r.tobytes()for r in x}),'cases':{k:{'T':len(v),'distinct_rows':len({r.tobytes()for r in v}),'SHA256':hashlib.sha256(v.tobytes()).hexdigest()}for k,v in inputs.items()},'not_current449_activations':True}
 model=Path('/shared/Kimi-K3');assert sha(model/'config.json')=='9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213'
 try:
  with mapped_layer_weights(model,'language_model.model.layers.1.block_sparse_moe.',b['layer_headers']['1'])as(held,desc,proof):
   assert len(held)==5384;report['mapping']=proof;save()
   payload={'config':json.loads((model/'config.json').read_bytes()),'weights':desc,'input_pointers':{k:int(v.ctypes.data)for k,v in inputs.items()},'node_rank':node,'master_addr':m['master_addr'],'master_port':m['master_port']}
   tick=time.monotonic();rc=lib.modern_kimi_TP32_moe_r163(json.dumps(payload).encode());report.update(returncode=rc,session_wall_s=time.monotonic()-tick,native=json.loads(lib.modern_kimi_TP32_moe_result_r163()));save()
   assert rc==0 and report['native']['completed']and report['native']['all8normal_before_communicators']
   owners=report['native']['reports'];assert len(owners)==8 and [o['global_tp_rank']for o in owners]==list(range(node*8,node*8+8))
   for o in owners:
    assert o['module_closed']and o['normal_Runtime_close']and len(o['arms'])==48
    for a in o['arms']:assert a['nonfinite']==0 and a['input_readonly_tail_canary']and a['private_provider_counter_delta_zero']and a['actual_I96_branch_hits']==int(a['candidate_I96'])
  del held,desc;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();report['memory_after']=memory();floor(report['memory_after']);report.update(completed=True,normal_CUDA_cleanup=True);save();print('KIMI_TP32_I96_R163_NORMAL',node,flush=True)
 except BaseException as e:report['error_class']=type(e).__name__;save();raise
if __name__=='__main__':run(sys.argv[1])
