"""Existing038d/5670 profile bool only; syntheticprefix512/allvisible, not Model criticalpath."""
from pathlib import Path
import ctypes,json,hashlib,os,time,traceback,gc
import torch,torch.multiprocessing as mp
from safetensors import safe_open
from plan import FIXTURES,records
from profile_contract import validate_profile
R=Path('/refactor');D=Path(__file__).resolve().parent;M=Path('/old/models/ZhipuAI-GLM-5.3');V=ctypes.c_void_p;Z=ctypes.c_size_t;I=ctypes.c_int
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();ptr=lambda t:V(t.data_ptr())
def verify():
 m=json.loads((D/'GPU_SOURCE_MANIFEST.json').read_bytes());assert sha(D/'GPU_SOURCE_MANIFEST.json')==os.environ['GLM370_GPU_SOURCE_SHA256']
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 p=R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json';assert sha(p)==m['Native_CPU_receipt_SHA256'];a=json.loads(p.read_bytes());assert a['completed']and a['CPU_only']and not a['dlopen']and not a['GPU_initialized']
 assert a['library_SHA256']==m['Native_library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'and sha(a['library'])==a['library_SHA256']
 for p,h in m['dependency_SHA256'].items():assert sha(p)==h,p
 return dict(library=a['library'],library_SHA256=a['library_SHA256']),m
def api(path):
 lib=ctypes.CDLL(path)
 lib.glm_layer_id_size_r5670.restype=Z;lib.glm_layer_id_r5670.argtypes=[V]
 lib.glm_layer_create_r5670.argtypes=[I,ctypes.c_char_p,V];lib.glm_layer_create_r5670.restype=V
 for name in ['error','report']:getattr(lib,'glm_layer_'+name+'_r5670').restype=ctypes.c_char_p
 lib.glm_layer_keys_r5670.argtypes=[V];lib.glm_layer_keys_r5670.restype=ctypes.c_char_p
 lib.glm_layer_load_r5670.argtypes=[V,ctypes.c_char_p]
 for name in ['finish','reset']:getattr(lib,'glm_layer_'+name+'_r5670').argtypes=[V]
 lib.glm_layer_forward_r5670.argtypes=[V,Z,Z,ctypes.POINTER(Z),V,V,V,I]
 lib.glm_layer_destroy_r5670.argtypes=[V,ctypes.POINTER(I)];lib.glm_layer_control_r5670.argtypes=[V,V]
 lib.modern_glm_dense_mode_r48_set.argtypes=[I]
 for name in ['modern_glm_vector_policy_r8540','modern_glm_down_row2_policy_r9284']:getattr(lib,name).argtypes=[I,I]
 lib.modern_glm_vector_counts_r8540.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z]
 return lib
def pci(device):
 lib=ctypes.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so');lib.cudaDeviceGetPCIBusId.argtypes=[ctypes.c_char_p,I,I];bus=ctypes.create_string_buffer(64);assert lib.cudaDeviceGetPCIBusId(bus,64,device)==0;return bus.value.decode()
def worker(rank,uid,global_start,pcis):
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(rank)
 cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=32;os.sched_setaffinity(0,cpus[rank*4:rank*4+4])
 torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 for name in ['allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']:setattr(torch.backends.cuda.matmul,name,False)
 a,m=verify();step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=R/'results'/('GLM370-'+step+'-rank%d.json'%rank);assert not path.exists()
 assert pci(rank)==pcis[rank]
 q=dict(completed=False,normal=False,rank=rank,PID=os.getpid(),actual_step=step,node=os.uname().nodename,physical_PCI=pcis[rank],device_count=torch.cuda.device_count(),current_device=torch.cuda.current_device(),CPU_affinity=sorted(os.sched_getaffinity(0)),Native_library_SHA256=a['library_SHA256'],GPU_source_manifest_SHA256=sha(D/'GPU_SOURCE_MANIFEST.json'),trials=[],warm=[],scope=__doc__,baseline='Direct original038d; vector and down-row2 OFF; only5670 profile bool changes',actual_Model_activation=False,Model_gain=False,layer=38,TP=8,N=1,B=1,synthetic_prefix_tokens=512,profile_mode='Mixed0/1 originalhook',measured_wall='Complete layer CAPI+original posttimer route readback+caller finalsync; Nativewall separately. Restore/prefix/weightload excluded equally. Not primitive timing.',profile_contract=m['profile_contract'])
 def save():
  t=path.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(path)
 save();lib=None;ctx=None;native_failed=False
 def check(rc):
  nonlocal native_failed
  if rc:
   native_failed=True;raise RuntimeError('Original Native failed, no reset/replay: '+repr(lib.glm_layer_error_r5670()))
 def limit():
  if time.time()>global_start+530:raise TimeoutError('Original single residency budget; no retry')
 try:
  lib=api(a['library']);check(lib.modern_glm_dense_mode_r48_set(2));check(lib.modern_glm_vector_policy_r8540(rank,0));check(lib.modern_glm_down_row2_policy_r9284(rank,0))
  class DlInfo(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',V),('sname',ctypes.c_char_p),('saddr',V)]
  process=ctypes.CDLL(None);process.dladdr.argtypes=[V,ctypes.POINTER(DlInfo)];process.dladdr.restype=I
  owners={}
  for sym in ['glm_layer_create_r5670','glm_layer_forward_r5670','glm_vec_gate_r8528','glm_grouped_gate_r5123','glm_grouped_down_r5123']:
   info=DlInfo();assert process.dladdr(ctypes.cast(getattr(lib,sym),V),ctypes.byref(info));p=Path(info.fname.decode()).resolve();expected=Path('/refactor/candidates/glm-hybrid-decode-r5386/libnative.so')if'grouped'in sym else Path(a['library']);assert p==expected.resolve();owners[sym]=dict(path=str(p),SHA256=sha(p))
  q['actual_symbol_owners']=owners
  cfg=json.loads((M/'config.json').read_bytes());cfg['probe_layers_r5680']=[38];q['checkpoint_config_SHA256']=sha(M/'config.json');q['checkpoint_index_SHA256']=sha(M/'model.safetensors.index.json')
  unique=ctypes.create_string_buffer(uid,len(uid));beg=time.monotonic();ctx=lib.glm_layer_create_r5670(rank,json.dumps(cfg).encode(),unique);assert ctx,repr(lib.glm_layer_error_r5670());q['create_seconds']=time.monotonic()-beg
  keys=json.loads(lib.glm_layer_keys_r5670(ctx));index=json.loads((M/'model.safetensors.index.json').read_bytes())['weight_map'];assert len(keys)==len(set(keys))and all(k.startswith('38.')and'model.layers.'+k in index for k in keys)
  beg=time.monotonic();referenced=0;handles={}
  try:
   for key in keys:
    limit();name=index['model.layers.'+key]
    if name not in handles:handles[name]=safe_open(str(M/name),framework='pt',device='cpu');handles[name].__enter__()
    value=handles[name].get_tensor('model.layers.'+key).contiguous();dtype=str(value.dtype);assert dtype in ['torch.float8_e4m3fn','torch.uint8','torch.float32','torch.bfloat16'];desc=dict(name=key,shape=list(value.shape),dtype=0 if value.dtype==torch.float32 else 1 if value.dtype==torch.bfloat16 else 2,pointer=value.data_ptr());check(lib.glm_layer_load_r5670(ctx,json.dumps([desc]).encode()));referenced+=value.nbytes
  finally:
   for h in handles.values():h.__exit__(None,None,None)
  check(lib.glm_layer_finish_r5670(ctx));q.update(weight_load_seconds=time.monotonic()-beg,weight_load_count=1,checkpoint_payload_referenced_bytes=referenced,state_dict_keys_SHA256=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),free_after_load=torch.cuda.mem_get_info(rank)[0]);assert q['free_after_load']>=6*1024**3;save()
  vote=torch.ones(1,device='cuda')
  def agree(ok):
   vote.fill_(float(ok));check(lib.glm_layer_control_r5670(ctx,ptr(vote)));assert vote.item()==8,'All8 numerical/state vote failed'
  def stats(name,n):
   v=(ctypes.c_uint64*n)();assert getattr(lib,name)(rank,v,n)==0;return list(v)
  prefix_pos=torch.arange(512,device='cuda',dtype=torch.int64);decode_pos=torch.tensor([512],device='cuda',dtype=torch.int64);pfdesc=(Z*3)(0,0,512);dcdesc=(Z*3)(0,512,1)
  prefix_guard=torch.full((512*6144+128,),123.,device='cuda',dtype=torch.bfloat16);prefix_out=prefix_guard[64:-64].view(512,6144)
  guard=torch.full((6144+128,),123.,device='cuda',dtype=torch.bfloat16);out=guard[64:-64].view(1,6144)
  for fixture in range(FIXTURES):
   gen=torch.Generator(device='cpu').manual_seed(367001+fixture);px=(torch.randn((512,6144),generator=gen,dtype=torch.bfloat16)*.1).cuda();x=(torch.randn((1,6144),generator=gen,dtype=torch.bfloat16)*.1).cuda();original=[p.clone()for p in [px,x,prefix_pos,decode_pos]]
   q.setdefault('input_fixtures',[]).append(dict(index=fixture,prefix_SHA256=hashlib.sha256(px.cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),decode_SHA256=hashlib.sha256(x.cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),synthetic=True));reference=None;own={}
   def call(mode,warm,pair,arm,rep):
    nonlocal reference
    limit();check(lib.glm_layer_reset_r5670(ctx));torch.cuda.synchronize(rank);check(lib.glm_layer_forward_r5670(ctx,512,1,pfdesc,ptr(px),ptr(prefix_pos),ptr(prefix_out),0))
    vector_before=stats('modern_glm_vector_counts_r8540',8);out.fill_(float('nan'));torch.cuda.synchronize(rank)
    start=time.perf_counter();check(lib.glm_layer_forward_r5670(ctx,1,1,dcdesc,ptr(x),ptr(decode_pos),ptr(out),mode));torch.cuda.synchronize(rank);wall=time.perf_counter()-start
    report=json.loads(lib.glm_layer_report_r5670());vector_after=stats('modern_glm_vector_counts_r8540',8);vd=[v-u for u,v in zip(vector_before,vector_after)];got=out.cpu().clone()
    validate_profile(report,mode)
    assert vd==[0,1,0,1,0,0,0,0],vd
    assert report['route_U']==8 and report['route_maxrows']==1
    if reference is None:assert mode==0;reference=got.clone()
    own.setdefault(mode,got.clone());diff=got.float()-reference.float();rel=float(diff.square().mean().sqrt())/max(float(reference.float().square().mean().sqrt()),1e-30);maxabs=float(diff.abs().max());repeat=torch.equal(got.view(torch.int16),own[mode].view(torch.int16));repeat_rel=float((got.float()-own[mode].float()).square().mean().sqrt())/max(float(own[mode].float().square().mean().sqrt()),1e-30);finite=bool(torch.isfinite(got).all());canary=bool((guard[:64]==123).all()and(guard[-64:]==123).all()and(prefix_guard[:64]==123).all()and(prefix_guard[-64:]==123).all());readonly=all(torch.equal(p.view(torch.uint8),old.view(torch.uint8))for p,old in zip([px,x,prefix_pos,decode_pos],original));free=torch.cuda.mem_get_info(rank)[0]
    agree(finite and canary and readonly and repeat_rel<=.005 and rel<=.005 and maxabs<=.05*max(1.,float(reference.float().abs().max()))and free>=512*1024**2)
    record=dict(fixture=fixture,mode=mode,warm=warm,pair=pair,arm=arm,rep=rep,complete_driver_wall_s=wall,whole_layer_native_wall_ms=report['wall_ms'],owner_threadCPU_ms=report['threadCPU_ms'],profile=mode,stages=report['stages'],lease_peak_logical_bytes=report['lease_peak_logical_bytes'],route_U=report['route_U'],route_maxrows=report['route_maxrows'],vector_counts_delta=vd,finite=finite,canary=canary,input_readonly=readonly,ownrepeat_bits_exact_diagnostic=repeat,ownrepeat_relative_RMS=repeat_rel,relative_RMS=rel,max_abs=maxabs,output_SHA256=hashlib.sha256(got.view(torch.uint8).numpy().tobytes()).hexdigest(),free_bytes=free)
    q['warm'if warm else'trials'].append(record);save()
   for record in records(fixture):call(**record)
  assert len(q['input_fixtures'])==1;assert json.loads(lib.glm_layer_keys_r5670(ctx))==keys
  check(lib.glm_layer_reset_r5670(ctx));expired=I();check(lib.glm_layer_destroy_r5670(ctx,ctypes.byref(expired)));ctx=None;assert expired.value==1
  del px,x,prefix_out,out,prefix_guard,guard;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize(rank);t=torch.ones(32,device='cuda');assert(t+1).sum().item()==64;torch.cuda.synchronize(rank)
  q.update(completed=True,normal=True,Runtime_weak_expired=True,normal_reset_destroy=True,Torch_after_destroy=True,policy_closed=True,free_after_release=torch.cuda.mem_get_info(rank)[0],measurement_calls=len(q['trials']),warm_calls=len(q['warm']));save();print('GLM370_NORMAL '+str(rank),flush=True)
 except BaseException as error:
  q.update(error=repr(error),traceback=traceback.format_exc(),Native_error_failstop_no_reset_or_replay=native_failed,normal=False);save();os._exit(2)
if __name__=='__main__':
 start=time.time();a,m=verify();assert torch.cuda.device_count()==8;torch.cuda.set_device(0);pcis=[pci(i)for i in range(8)];assert len(set(pcis))==8;lib=api(a['library']);size=lib.glm_layer_id_size_r5670();assert 0<size<1024;uid=ctypes.create_string_buffer(size);assert lib.glm_layer_id_r5670(uid)==0
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];p=R/'results'/('GLM370-'+step+'-parent.json')
 with p.open('x')as f:json.dump(dict(parent_PID=os.getpid(),actual_step=step,GPU_source_manifest_SHA256=sha(D/'GPU_SOURCE_MANIFEST.json'),original_start_unix=start,unique_physical_PCIs=pcis),f)
 mp.spawn(worker,args=(uid.raw,start,pcis),nprocs=8,join=True)
 rows=[json.loads((R/'results'/('GLM370-'+step+'-rank%d.json'%rank)).read_bytes())for rank in range(8)];assert all(r['normal']and r['completed']and r['measurement_calls']==12 and r['warm_calls']==2 for r in rows)
 print('GLM370_ALL8_NORMAL',flush=True)
