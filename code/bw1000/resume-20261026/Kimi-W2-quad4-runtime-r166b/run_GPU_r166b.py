"""Root-authorized finiteL1 component only; no Slurm or source mutation."""
from pathlib import Path
import ctypes,gc,hashlib,json,os,sys,socket,re,time
import numpy as np
import torch
D=Path(__file__).resolve().parent;C=Path('/refactor/incoming/Kimi-W2-quad4-component-r166');R=Path('/refactor');sys.path.insert(0,str(C))
from Kimi_full_MoE_input_r8669s import load_fixture
from mapped_layer_weights_r9125 import mapped_layer_weights
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(mp):
 m=json.loads(Path(mp).read_bytes());assert m['GPU_authorized']is True and m['source_published']is True and m['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',m['publication_commit'])
 assert m['parent_job']=='910374'and os.environ['SLURM_JOB_ID']==m['parent_job']and int(os.environ['SLURM_NTASKS'])==2
 assert m['expected_nodes']==['a02r4n01','a02r4n04'];node=int(os.environ['SLURM_PROCID']);assert node in [0,1]and socket.gethostname()==m['expected_nodes'][node]
 assert sha(D/'runtime-source-manifest-r166b.json')==m['runtime_source_manifest_SHA256']
 for n,h in json.loads((D/'runtime-source-manifest-r166b.json').read_bytes())['files_SHA256'].items():assert sha(D/n)==h
 assert sha(C/'source-manifest-r166.json')==m['CPU_source_manifest_SHA256'];source=json.loads((C/'source-manifest-r166.json').read_bytes())
 for n,h in source['files_SHA256'].items():assert sha(C/n)==h
 bp=Path(m['CPU_build_path']);assert sha(bp)==m['CPU_build_SHA256'];b=json.loads(bp.read_bytes());assert b['completed']and b['CPU_only']and not b['GPU']and not b['dlopen']and b['source_manifest_SHA256']==sha(C/'source-manifest-r166.json')
 for k in ['native','library']:assert sha(b[k])==b[k+'_SHA256']
 ctx=json.loads((C/'parent-GPU-context-r166.json').read_bytes())
 for p,h in ctx['pinned_libraries'].items():assert sha(p)==h
 for p,h in ctx['network_library_files'].items():assert sha(p)==h
 torch.set_num_threads(1);assert torch.cuda.device_count()==8 and len(os.sched_getaffinity(0))>=32;torch.cuda.set_device(0)
 lib=ctypes.CDLL(b['library']);lib.modern_kimi_TP16_quad_run_r166.argtypes=[ctypes.c_char_p];lib.modern_kimi_TP16_quad_run_r166.restype=ctypes.c_int;lib.modern_kimi_TP16_quad_result_r166.restype=ctypes.c_char_p
 class DI(ctypes.Structure):_fields_=[('name',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
 dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DI)];bindings=[]
 for name,expected in [(n,b['native_SHA256'])for n in ['modern_kimi_grouped_I192_r8669y','modern_kimi_grouped_I192_parent503_r166','modern_kimi_W2_quad_counts_r166','modern_kimi_W2_event_arm_r166','modern_kimi_W2_event_read_r166','modern_kimi_W2_event_close_r166','modern_kimi_moe_event_arm_r9220','modern_kimi_moe_phase_mark_r155']]+[('modern_cpu_copy_stats_r2515',ctx['Runtime_SHA256'])]:
  z=DI();assert dl.dladdr(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(z))and sha(z.name.decode())==expected;bindings.append({'symbol':name,'actual_DSO_SHA256':expected})
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];outpath=R/'results'/('Kimi-W2-quad4-r166-'+step+'-node%d.json'%node);assert not outpath.exists()
 out={'completed':False,'actual_step':step,'node':node,'actual_bindings':bindings,'manifest_SHA256':sha(mp),'CPU_build_SHA256':sha(bp),'scope':'realL1TP16 controlledT8history/W8geometrynotC1traffic,T64historyandGaussian;notModeloractual449routes','experiments':[]}
 def save():tmp=outpath.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+chr(10));tmp.replace(outpath)
 save();raw,fixture=load_fixture(R,ctx['fixture']);history=np.frombuffer(raw,np.uint16).reshape(16,7168);first8=history[:8].copy();repeat=np.tile(history,(4,1)).copy();f=(history.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f).all();rms=float(np.sqrt(np.mean(f.astype(np.float64)**2)));rng=np.random.default_rng(15520261028);noise=(rng.standard_normal((64,7168))*rms).astype(np.float32);u=noise.view(np.uint32);random=((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16);inputs=[first8,repeat,random]
 model=Path('/shared/Kimi-K3');assert sha(model/'config.json')=='9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213'
 try:
  with mapped_layer_weights(model,'language_model.model.layers.1.block_sparse_moe.',b['layer_headers']['1'])as(held,descriptors,proof):
   assert len(held)==5384
   e={'layer':1,'mapping':proof,'fixture':fixture,'workloads':[{'index':i,'T':len(a),'name':kind,'input_SHA256':hashlib.sha256(a.tobytes()).hexdigest(),'distinct_rows':len({r.tobytes()for r in a})}for i,(kind,a)in enumerate(zip(['T8historicalfirst8_W8geometrynotC1traffic','T64historical16repeat4','T64controlledGaussianSameRMS_seed155'],inputs))]};out['experiments'].append(e);save()
   payload={'config':json.loads((model/'config.json').read_bytes()),'layer':1,'weights':descriptors,'input_pointers':{str(i):int(a.ctypes.data)for i,a in enumerate(inputs)},'node_rank':node,'master_addr':m['master_addr'],'master_port':m['master_port']}
   tick=time.monotonic();rc=lib.modern_kimi_TP16_quad_run_r166(json.dumps(payload).encode());e.update(returncode=rc,wall_s=time.monotonic()-tick,native=json.loads(lib.modern_kimi_TP16_quad_result_r166()));save();assert rc==0 and e['native']['completed']and e['native']['all8normal_before_communicators']
   for owner in e['native']['reports']:
    assert owner['normal_Runtime_close']and owner['module_closed']and len(owner['arms'])==36 and len(owner['profile'])==16 and len(owner['W2_event_samples'])==24
    assert all(a['forward_finite']and a['NRMSE_vs_first_warm']<=.005 and a['input_readonly_and_tail_canary']for a in owner['arms'])
    for p in owner['profile']:assert p['T']==64 and sum(p['expert_hist'])==1024
    for p in owner['W2_event_samples']:assert p['W2_stream_ms']>0 and p['T']in[8,64]
  gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();assert min(torch.cuda.mem_get_info(i)[0]for i in range(8))>4*1024**3
  out.update(completed=True,normal8_Runtime_owners=True,normal_CUDA_cleanup=True);save();print('KIMI_W2_QUAD_R166_NORMAL',node,flush=True)
 except BaseException as error:out['error_class']=type(error).__name__;save();raise
if __name__=='__main__':main(sys.argv[1])
