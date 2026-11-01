"""Known Runtime baseline first; complete T1 HC pre, independent eight fixtures."""
import pathlib,json,hashlib,os,time,ctypes,traceback,statistics,numpy as np
from fixture import load
from affinity_protocol import rank_plan
R=pathlib.Path('/refactor');S=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
rank=int(os.environ['SLURM_LOCALID']);initial_affinity,assigned_affinity=rank_plan(rank);step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/f'Deep358-GPU-r1-{step}-rank{rank}.json';assert not out.exists()
q=dict(completed=False,passed=False,actual_step=step,rank=rank,Model_loaded=False,scope='Actual afc6 HyperConnection::pre versus T1 epilogue fusion; actualHC parameters/synthetic residual only. No Model throughput or quality claim.',calls=[])
def save():out.write_text(json.dumps(q,indent=2)+'\n')
begin=time.perf_counter();ctx=None;lib=None
try:
 mfile=S/'GPU_MANIFEST.json';assert sha(mfile)==os.environ['DEEP358_MANIFEST_SHA'];m=json.loads(mfile.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 bp=R/'candidates/Deep-HC-epilogue-r358-cpu-r1/CPU-build.json';assert sha(bp)==m['CPU_receipt_SHA256'];build=json.loads(bp.read_bytes());assert build['completed']and build['CPU_only']and not build['GPU_initialized']and not build['Native_dlopen']
 library=pathlib.Path(build['library']);assert sha(library)==build['library_SHA256']and sha(build['Native_path'])==build['Native_SHA256']=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
 import torch
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(rank)
 props=torch.cuda.get_device_properties(rank);assert 'gfx936'in getattr(props,'gcnArchName','');q['hardware']=dict(node=os.environ['SLURMD_NODENAME'],cards=1,visible_physical_HIP=str(rank),visible_device_list=os.environ['HIP_VISIBLE_DEVICES'],Runtime_device=rank,name=props.name,arch=props.gcnArchName)
 data,proof=load(rank);fp=bp.with_name('CPU-fixtures.json');assert sha(fp)==build['CPU_fixture_receipt_SHA256'];fixtures=json.loads(fp.read_bytes());assert fixtures['passed']and not fixtures['GPU_initialized']and not fixtures['Native_dlopen'];old=fixtures['fixtures'][rank];assert proof['weights']==old['weights']and proof['input_SHA256']==old['input_SHA256']and proof['config_SHA256']==old['config_SHA256'];q['fixture']=proof;q['CPU_affinity_assigned']=assigned_affinity;q['CPU_receipt_SHA256']=sha(bp);q['library_SHA256']=sha(library);q['memory_before_context']=list(torch.cuda.mem_get_info())
 lib=ctypes.CDLL(str(library));P=ctypes.c_void_p;I=ctypes.c_int;U=ctypes.c_uint;D=ctypes.c_double
 for n in ['error','report']:getattr(lib,'deep_hc_epilogue_'+n+'_r358').restype=ctypes.c_char_p
 lib.deep_hc_epilogue_create_r358.argtypes=[I,D,D,I,P,P,P];lib.deep_hc_epilogue_create_r358.restype=P
 lib.deep_hc_epilogue_forward_r358.argtypes=[P,I,ctypes.POINTER(P),P];lib.deep_hc_epilogue_forward_r358.restype=I
 lib.deep_hc_epilogue_destroy_r358.argtypes=[P,ctypes.POINTER(I)];lib.deep_hc_epilogue_hit_r358.argtypes=[P,ctypes.POINTER(U)]
 lib.deep_hc_epilogue_state_r358.argtypes=[P,ctypes.POINTER(P)]
 def check(rc):
  if rc:raise RuntimeError(lib.deep_hc_epilogue_error_r358().decode())
 lib.deep_hc_canonical_calls_r7130.argtypes=[I,I];lib.deep_hc_canonical_calls_r7130.restype=ctypes.c_uint64
 weights=data['weights'];ctx=lib.deep_hc_epilogue_create_r358(rank,data['rms_eps'],data['hc_eps'],data['iterations'],*[P(a.ctypes.data)for a in weights]);assert ctx,lib.deep_hc_epilogue_error_r358().decode()
 q['CPU_affinity_initial']=initial_affinity;q['CPU_affinity_after_Runtime_create']=sorted(os.sched_getaffinity(0));assert q['CPU_affinity_after_Runtime_create']in [initial_affinity,assigned_affinity]
 backing=[torch.empty(16384+64,device=f'cuda:{rank}',dtype=torch.bfloat16),torch.empty(4096+64,device=f'cuda:{rank}',dtype=torch.bfloat16),torch.empty(4+16,device=f'cuda:{rank}',dtype=torch.float32),torch.empty(16+16,device=f'cuda:{rank}',dtype=torch.float32)]
 x=backing[0][:16384].view(1,4,4096);x.copy_(torch.from_numpy(data['input_bits']).view(torch.bfloat16).cuda())
 outputs=[backing[1][:4096].view(1,4096),backing[2][:4].view(1,4),backing[3][:16].view(1,4,4)];buffers=[x,*outputs];ptrs=(P*4)(*[P(t.data_ptr())for t in buffers])
 tails=[backing[0][16384:],backing[1][4096:],backing[2][4:],backing[3][16:]]
 for tail in tails:tail.fill_(17.)
 def state():
  arrays=[np.empty_like(a)for a in weights]+[np.empty_like(a)for a in weights]+[np.empty(3,np.float32)]
  pointers=(P*7)(*[P(a.ctypes.data)for a in arrays]);check(lib.deep_hc_epilogue_state_r358(ctx,pointers))
  assert all(np.array_equal(a,b)for a,b in zip(arrays[:3],weights))and all(np.array_equal(a,b)for a,b in zip(arrays[3:6],weights));assert np.array_equal(arrays[6],np.array([data['rms_eps'],data['hc_eps'],2],np.float32))
  return [hashlib.sha256(a.tobytes()).hexdigest()for a in arrays]
 q['weights_state_initial']=state()
 def numeric():
  actual=[outputs[0].cpu().view(torch.uint16).numpy().reshape(4096).copy(),outputs[1].cpu().numpy().reshape(4).copy(),outputs[2].cpu().numpy().reshape(4,4).copy()];metrics=[]
  for name,a in zip(['collapsed_BF16','post','comb'],actual):
   b=data['expected'][name];av=(a.astype(np.uint32)<<16).view(np.float32)if a.dtype==np.uint16 else a;bv=(b.astype(np.uint32)<<16).view(np.float32)if b.dtype==np.uint16 else b;assert np.isfinite(av).all()and np.isfinite(bv).all();delta=av.astype(np.float64)-bv.astype(np.float64);nrmse=float(np.linalg.norm(delta)/(np.linalg.norm(bv.astype(np.float64))+1e-30));assert nrmse<(.005 if name=='collapsed_BF16'else 1e-4),(name,nrmse)
   metrics.append(dict(name=name,NRMSE=nrmse,max_abs=float(np.max(np.abs(delta))),exact_diagnostic=bool(np.array_equal(a,b))))
  assert all(bool(torch.all(t==17.))for t in tails);assert np.array_equal(x.cpu().view(torch.uint16).numpy(),data['input_bits'])
  return metrics
 def call(mode,events=False):
  os.environ['INFINILM_DEEP_HC_EPILOGUE_R358']=str(mode)
  for t in outputs:t.fill_(float('nan'))
  torch.cuda.synchronize();start_event=end_event=None
  if events:start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True);start_event.record()
  baseline_before=int(lib.deep_hc_canonical_calls_r7130(rank,2));tick=time.perf_counter();check(lib.deep_hc_epilogue_forward_r358(ctx,mode,ptrs,P(torch.cuda.current_stream().cuda_stream)));wall=time.perf_counter()-tick
  if events:end_event.record();end_event.synchronize()
  baseline_hits=int(lib.deep_hc_canonical_calls_r7130(rank,2))-baseline_before;assert baseline_hits==1-mode
  detail=json.loads(lib.deep_hc_epilogue_report_r358().decode());hit=U();check(lib.deep_hc_epilogue_hit_r358(ctx,ctypes.byref(hit)));assert detail['candidate']==bool(mode)and hit.value==mode
  return dict(mode=mode,CAPI_wall_s=wall,native=detail,GPU_hit=hit.value,actual_afc6_baseline_hits=baseline_hits,producer_stream_event_ms=start_event.elapsed_time(end_event)if events else None,numerics=numeric())
 # No candidate may run until the independent CPU full-chain baseline check passes.
 q['baseline_numeric_first']=call(0,True);q['CPU_affinity_after_first_baseline']=sorted(os.sched_getaffinity(0));assert q['CPU_affinity_after_first_baseline']==assigned_affinity;save();q['candidate_numeric']=call(1,True);save()
 q['warmup']=[call(0),call(1)];q['prepare_and_qualification_s']=time.perf_counter()-begin;measurement=time.perf_counter()
 for trial,mode in enumerate([0,1,1,0]*5):
  entries=[call(mode)for _ in range(32)];q['calls'].append(dict(trial=trial,mode=mode,repeats=32,CAPI_wall_s=sum(v['CAPI_wall_s']for v in entries),native_component_s=sum(v['native']['HCpre_staging_outputcopy_drained_seconds']for v in entries),GPU_hits=sum(v['GPU_hit']for v in entries),max_NRMSE=max(z['NRMSE']for v in entries for z in v['numerics'])));save()
 q['measurement_loop_wall_s']=time.perf_counter()-measurement;q['actual_CAPI_measurement_s']=sum(v['CAPI_wall_s']for v in q['calls']);q['weights_state_final']=state();assert q['weights_state_initial']==q['weights_state_final']
 q['pairs']=[]
 for i in range(0,20,4):
  a,b,c,d=q['calls'][i:i+4];off=(a['CAPI_wall_s']+d['CAPI_wall_s'])/2;on=(b['CAPI_wall_s']+c['CAPI_wall_s'])/2;q['pairs'].append(dict(ON_over_OFF=on/off,OFF_endpoint_drift=d['CAPI_wall_s']/a['CAPI_wall_s']-1,OFF_s=off,ON_s=on))
 q.update(passed=True,completed=True,memory_after_measurement=list(torch.cuda.mem_get_info()),event_scope='Only qualification: producer-stream events bracket complete synchronous CAPI; includes waits/hostlaunch gaps,notkernel-only. Noevents inABBA.',launch_scope='Candidate explicitT1 oneblock256, launch_bounds256, GPUhit1/call. Originalnorm/Gemm/Sinkhorn unchanged; highlevelsubmits are not actualSDKkernelcounts.',quality_scope='Component finite/numeric/state checks only; noModelfinaltokens/QA orspeed promotion')
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc(),completed=True)
finally:
 if ctx:
  expired=ctypes.c_int();q['destroy_code']=lib.deep_hc_epilogue_destroy_r358(ctx,ctypes.byref(expired));q['Runtime_expired']=expired.value;q['normal_close']=q['destroy_code']==0 and expired.value==1
 q['overall_wall_s']=time.perf_counter()-begin;save()
assert q['passed']and q.get('normal_close'),q.get('error')
