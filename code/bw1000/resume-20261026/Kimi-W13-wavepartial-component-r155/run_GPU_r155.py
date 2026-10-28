"""TP16 component only; layers loaded sequentially, never a full-model replay."""
from pathlib import Path
import ctypes,gc,hashlib,json,os,sys,time,re,socket
import numpy as np
import torch
from Kimi_full_MoE_input_r8669s import load_fixture
from mapped_layer_weights_r9125 import mapped_layer_weights
D=Path(__file__).resolve().parent;R=Path('/refactor');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(manifest_path):
 m=json.loads(Path(manifest_path).read_bytes());assert m['GPU_authorized']is True and m['source_published']is True
 assert m['publication_branch']=='bw1000-experiments' and re.fullmatch('[0-9a-f]{40}',m['publication_commit'])
 assert os.environ['SLURM_JOB_ID']==m['parent_job'] and int(os.environ['SLURM_NTASKS'])==2 and len(m['expected_nodes'])==2
 assert socket.gethostname()==m['expected_nodes'][int(os.environ['SLURM_PROCID'])]
 build_path=Path(m['CPU_build_path']);assert sha(build_path)==m['CPU_build_SHA256'];b=json.loads(build_path.read_bytes());assert b['completed']and b['CPU_only']and not b['GPU']
 assert sha(D/'source-manifest-r155.json')==b['source_manifest_SHA256'];sources=json.loads((D/'source-manifest-r155.json').read_bytes())
 for rel,h in sources['files_SHA256'].items():assert sha(D/rel)==h
 assert sha(b['native'])==b['native_SHA256']and sha(b['library'])==b['library_SHA256']
 ctx=json.loads((D/'parent-GPU-context-r155.json').read_bytes())
 for p,h in ctx['pinned_libraries'].items():assert sha(p)==h
 for p,h in ctx['network_library_files'].items():assert sha(p)==h
 torch.set_num_threads(1);assert torch.cuda.device_count()==8 and len(os.sched_getaffinity(0))>=32;torch.cuda.set_device(0)
 lib=ctypes.CDLL(b['library']);lib.modern_kimi_TP16_moe_r8669w.argtypes=[ctypes.c_char_p];lib.modern_kimi_TP16_moe_r8669w.restype=ctypes.c_int;lib.modern_kimi_TP16_moe_result_r8669w.restype=ctypes.c_char_p
 class DI(ctypes.Structure):_fields_=[('name',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
 dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DI)];bindings=[]
 for name,expected in [(n,b['native_SHA256'])for n in ['modern_kimi_moe_event_arm_r9220','modern_kimi_moe_phase_mark_r155','modern_kimi_W13_wavepartial_counts_r155','modern_kimi_grouped_I192_r8669y','modern_kimi_I192_T_counts_r9129']]+[('modern_cpu_copy_stats_r2515',ctx['Runtime_SHA256'])]:
  z=DI();assert dl.dladdr(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(z));assert sha(z.name.decode())==expected;bindings.append({'symbol':name,'actual_DSO_SHA256':sha(z.name.decode())})
 node=int(os.environ['SLURM_PROCID']);assert node in [0,1];path=R/'results'/('Kimi-MoE-phase-r155-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'-node%d.json'%node);assert not path.exists()
 out={'completed':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':node,'actual_bindings':bindings,'experiments':[],'scope':'controlled TP16 two real MoE layers; historical-repeat and Gaussian-control, NOT current445 activation/routes or PP critical chain'}
 def save():tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(path)
 save();raw,fixture=load_fixture(R,ctx['fixture']);x=np.frombuffer(raw,np.uint16).reshape(16,7168);repeat=np.tile(x,(4,1)).copy();f=(x.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f).all();rms=float(np.sqrt(np.mean(f.astype(np.float64)**2)));assert rms>0
 rng=np.random.default_rng(15520261028);noise=(rng.standard_normal((64,7168))*rms).astype(np.float32);u=noise.view(np.uint32);random=((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)
 model=Path('/shared/Kimi-K3');assert sha(model/'config.json')=='9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213'
 try:
  for layer in [1,47]:
   with mapped_layer_weights(model,'language_model.model.layers.%d.block_sparse_moe.'%layer,b['layer_headers'][str(layer)])as(held,descriptors,proof):
    assert len(held)==5384
    row={'layer':layer,'input_current445':False,'mapping':proof,'fixture':fixture,'workloads':[{'index':i,'name':kind,'input_SHA256':hashlib.sha256(a.tobytes()).hexdigest(),'distinct_rows':len({r.tobytes()for r in a})}for i,(kind,a)in enumerate([('historical16_repeated64',repeat),('controlled_Gaussian64_same_RMS',random)])]};out['experiments'].append(row);save()
    q={'config':json.loads((model/'config.json').read_bytes()),'layer':layer,'weights':descriptors,'input_pointers':{'0':int(repeat.ctypes.data),'1':int(random.ctypes.data)},'node_rank':node,'master_addr':m['master_addr'],'master_port':int(m['master_port'])+layer*4}
    tick=time.monotonic();rc=lib.modern_kimi_TP16_moe_r8669w(json.dumps(q).encode());row.update(returncode=rc,wall_s=time.monotonic()-tick,native=json.loads(lib.modern_kimi_TP16_moe_result_r8669w()));save();assert rc==0 and row['native']['completed']and row['native']['all8normal_before_communicators']
    for owner in row['native']['reports']:
     assert owner['normal_Runtime_close']and owner['module_closed']and len(owner['arms'])==26 and len(owner['profile'])==18
     for profile in owner['profile']:
      assert sum(profile['expert_hist'])==1024 and profile['T']==64 and profile['layer']==layer
      profile['TT2_jobs']=sum((x+1)//2 for x in profile['expert_hist']);profile['TT2_partial_jobs']=sum(x%2 for x in profile['expert_hist']);profile['expert_unique']=sum(x>0 for x in profile['expert_hist']);profile['max_expert_rows']=max(profile['expert_hist'])
     assert all(a['forward_finite']and a['NRMSE_vs_first_warm']<=0.005 and a['input_readonly_and_tail_canary']for a in owner['arms'])
    save()
   gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();assert min(torch.cuda.mem_get_info(i)[0]for i in range(8))>4*1024**3
  out.update(completed=True,normal8_Runtime_owners_per_experiment=True,normal_CUDA_cleanup=True);save();print('KIMI_MOE_PHASE_R150_NORMAL',node,flush=True)
 except BaseException as error:out['error_class']=type(error).__name__;save();raise
if __name__=='__main__':run(sys.argv[1])
