"""Fixed synthetic RMS/RoPE and actual source-map falsification. No Model/weights."""
import pathlib,json,hashlib,ctypes,os,time,threading,concurrent.futures
from Deep_latent_maps_r8946 import maps,validate
import numpy as np
import resource
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=r/'results'/('deep-host-clock-r9049-'+tag+'.json');assert not path.exists();bp=r/'results/deep-scalar-Cast-CPU-r9054.json';assert sha(bp)=='3cdd830cf12ac295b764ef7fa09c9698c425c4be85643b02cfed14d78e99a109';cpu=json.loads(bp.read_bytes());assert cpu['completed']and cpu['CPU_only'];artifact=cpu['libraries']['full']['path'];assert sha(artifact)=='59340bfa4720170c7f57f7178d3eb74fb7e9b19409cdf47b6223d9ca439221f0';m=dict(DSO_SHA256=cpu['libraries']['full']['SHA256'])
for n,h in cpu['link']['full']['objects'].items():assert sha(n)==h
for n,v in cpu['actual_DTK_bindings'].items():assert sha(n)==v['SHA256']
import torch
torch.set_num_threads(1);assert torch.cuda.device_count()==8;lib=ctypes.CDLL(artifact);fn=lib.modern_deep_latent_strided_r681;fn.argtypes=[ctypes.c_int]+[ctypes.c_size_t]*6+[ctypes.c_int64]*8+[ctypes.c_void_p]*9;fn.restype=ctypes.c_int;lib.modern_deep_latent_error.restype=ctypes.c_char_p;lib.modern_deep_latent_shutdown.restype=ctypes.c_int;lib.deep_latent_prefill_mode_r8933.argtypes=[ctypes.c_int];lib.deep_reference_scratch_mode_r8953.argtypes=[ctypes.c_int];lib.deep_reference_scratch_mode_r8953.restype=ctypes.c_int;lib.deep_reference_packed_batch_mode_r8996b.argtypes=[ctypes.c_int];lib.deep_reference_packed_batch_mode_r8996b.restype=ctypes.c_int;lib.deep_reference_packed_batch_counts_r8996b.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.deep_reference_packed_batch_counts_r8996b.restype=ctypes.c_int;lib.deep_reference_scratch_counts_r8954.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.deep_reference_scratch_counts_r8954.restype=ctypes.c_int
lib.deep_sink_broadcast_mode_r9040.argtypes=[ctypes.c_int];lib.deep_sink_broadcast_mode_r9040.restype=ctypes.c_int;lib.deep_sink_broadcast_counts_r9040.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.deep_sink_broadcast_counts_r9040.restype=ctypes.c_int
lib.deep_scalar_cast_mode_r9054.argtypes=[ctypes.c_int];lib.deep_scalar_cast_mode_r9054.restype=ctypes.c_int;lib.deep_scalar_cast_counts_r9054.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.deep_scalar_cast_counts_r9054.restype=ctypes.c_int
report=dict(hardware='BW1000',level='PACKED_EQUAL_N_ATTENTION_COMPONENT',status='RUNNING',completed=False,step=tag,DSO_SHA256=m['DSO_SHA256'],fixtures=[],normal_owners=[],CPU_clock_limit='OSthreadCPUincludesHIPspin/enqueue;notpureCPUoperatorworkorGPUbusy;timingcallsoutsideCAPIonly,noC++/mathchange',scope='Current1000mode2+sinkON FIXED; scalarprefix Cast OFF/ON only, defaultOFF; oneHOST/noGPUnewkernel. Actualselectionmaps, syntheticRMSRoPE QKV not saved1000activation. Original2DQKPV/Softmax/sinkCopy/drain preserved;R127 tail1 boundaries,2MiB scratch. 2warm6timedOFFONONOFFOFFONrestoreOFF; noModel/nopeak/no15tokclaim.')
PLAN=[dict(phase='warm',arm=v)for v in [0,1]]+[dict(phase='measure',arm=v)for v in [0,1,1,0,0,1]]+[dict(phase='restore',arm=0)]
EXPECTED={'swa':[1,0,508,4,1540,2048,3],'hca128':[1,0,512,12,1536,2096,3],'csa4':[1,0,512,384,1536,3584,3]}
state={};pools=[concurrent.futures.ThreadPoolExecutor(max_workers=1)for _ in range(8)]
def save():
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
def check(code):
 if code:raise RuntimeError(lib.modern_deep_latent_error().decode())
def counts(d):
 q=(ctypes.c_uint64*4)();check(lib.deep_reference_scratch_counts_r8954(d,q,4));return list(q)
def setup(d,kind):
 torch.cuda.set_device(d);gen=torch.Generator(device='cuda:'+str(d));seed=894600+100*d+['swa','hca128','csa4'].index(kind);gen.manual_seed(seed)
 def norm(shape,rope=False):
  x=torch.randn(shape,device='cuda',generator=gen,dtype=torch.float32);x=x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()
  if rope:
   pos=(torch.arange(shape[0],device='cuda')%512).float();freq=torch.pow(10000.,-torch.arange(0,64,2,device='cuda').float()/64);angle=pos[:,None]*freq[None,:];a=x[...,-64::2].clone();b=x[...,-63::2].clone();co=angle.cos();sn=angle.sin()
   if len(shape)==3:co=co[:,None,:];sn=sn[:,None,:]
   x[...,-64::2]=a*co-b*sn;x[...,-63::2]=a*sn+b*co
  return x.to(torch.bfloat16)
 si,ci,lens,cr,ratio=maps(kind);q=norm((2048,8,512),True);s=norm((2048,512),True);c=norm((cr,512));sink=torch.linspace(-.5,.5,8,device='cuda',dtype=torch.float32)
 args=(q,s,c,sink,torch.from_numpy(si).cuda(),torch.from_numpy(ci).cuda(),torch.from_numpy(lens).cuda());backs=[torch.full((q.numel()+128,),42.,device=q.device,dtype=q.dtype)for _ in PLAN];outs=[b[:q.numel()].view_as(q)for b in backs];torch.cuda.synchronize();sample=q.flatten()[:128].view(torch.int16).cpu().numpy().tobytes();state[d]=dict(args=args,outs=outs,backs=backs,seed=seed,thread=threading.get_native_id(),cw=ci.shape[1],cr=cr,kind=kind,input_clones=[v.clone()for v in args])
 return dict(device=d,seed=seed,operand_sample_SHA256=hashlib.sha256(sample).hexdigest(),query_RMS=float(q.float().square().mean().sqrt()),KV_RMS=float(s.float().square().mean().sqrt()),stream=int(torch.cuda.current_stream().cuda_stream))
def invoke(d,which):
 width=2048;arm=PLAN[which]['arm']
 z=state[d];assert threading.get_native_id()==z['thread'];torch.cuda.set_device(d);q,s,c,sink,si,ci,lens=z['args'];out=z['outs'][which];before=counts(d);uc_before=(ctypes.c_uint64*7)();check(lib.deep_reference_packed_batch_counts_r8996b(d,uc_before,7));sc_before=(ctypes.c_uint64*3)();check(lib.deep_scalar_cast_counts_r9054(d,sc_before,3));bc_before=(ctypes.c_uint64*3)();check(lib.deep_sink_broadcast_counts_r9040(d,bc_before,3));usage_before=resource.getrusage(resource.RUSAGE_THREAD);thread_before=time.thread_time();tick=time.perf_counter()
 for a in range(0,2048,width):
  qs=q[a:a+width];ss=si[a:a+width];cs=ci[a:a+width];ls=lens[a:a+width];dst=out[a:a+width];ptr=lambda x:ctypes.c_void_p(x.data_ptr());args=[d,width,8,2048,z['cr'],128,z['cw'],*qs.stride(),*s.stride(),*c.stride(),sink.stride(0),*[ptr(x)for x in [qs,s,c,sink,ss,cs,ls]],ptr(dst),ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)];check(fn(*args))
 elapsed=time.perf_counter()-tick;thread_cpu=time.thread_time()-thread_before;usage_after=resource.getrusage(resource.RUSAGE_THREAD);after=counts(d);delta=[y-x for x,y in zip(before,after)];assert delta==[0,1,0,3];uc_after=(ctypes.c_uint64*7)();check(lib.deep_reference_packed_batch_counts_r8996b(d,uc_after,7));uc_delta=[y-x for x,y in zip(uc_before,uc_after)];expected=list(EXPECTED[z['kind']]);expected[5]+=2*expected[4]-2*expected[3];expected[5]-=504 if arm else 0;assert uc_delta==expected,(arm,uc_delta,expected);bc_after=(ctypes.c_uint64*3)();check(lib.deep_sink_broadcast_counts_r9040(d,bc_after,3));bc_delta=[y-x for x,y in zip(bc_before,bc_after)];bc_expected=[expected[3],8*expected[4],expected[4]-expected[3]];assert bc_delta==bc_expected,(bc_delta,bc_expected);sc_after=(ctypes.c_uint64*3)();check(lib.deep_scalar_cast_counts_r9054(d,sc_after,3));sc_delta=[y-x for x,y in zip(sc_before,sc_after)];assert sc_delta==([4,508,504]if arm else[0,0,0]);assert all(torch.equal(v,c)for v,c in zip(z['args'],z['input_clones']));assert torch.isfinite(out.float()).all();assert bool((z['backs'][which][-128:]==42).all());return dict(device=d,seconds=elapsed,thread_CPU_seconds=thread_cpu,thread_CPU_fraction=thread_cpu/elapsed,thread_user_seconds=usage_after.ru_utime-usage_before.ru_utime,thread_system_seconds=usage_after.ru_stime-usage_before.ru_stime,voluntary_context_switches=usage_after.ru_nvcsw-usage_before.ru_nvcsw,involuntary_context_switches=usage_after.ru_nivcsw-usage_before.ru_nivcsw,counter_delta=delta,packed_counter_delta=uc_delta,broadcast_counter_delta=bc_delta,scalar_cast_counter_delta=sc_delta,inputs_readonly=True,CAPI_calls=2048//width,finite=True,canary=True)
def compare(d,a,b):
 x=state[d]['outs'][a].float();y=state[d]['outs'][b].float();delta=(y-x).abs();row=delta.amax((1,2));bits=lambda z:z.view(torch.int16).to(torch.int32)&65535
 def ordered(z):
  u=bits(z);return torch.where((u&32768)!=0,65535-u,u+32768)
 ulp=(ordered(state[d]['outs'][a])-ordered(state[d]['outs'][b])).abs();mask=delta!=0;tile=(torch.arange(2048,device='cuda')%32==0);request=(torch.arange(2048,device='cuda')%512==0)
 return dict(device=d,reference=a,candidate=b,maxabs=float(delta.max()),relativeRMS=float(delta.square().mean().sqrt()/x.square().mean().sqrt().clamp_min(1e-9)),ULP_max=int(ulp.max()),ULP_mean=float(ulp.float().mean()),nonzero_elements=int(mask.sum()),rows_with_difference=int((row!=0).sum()),tile_start_row_maxabs=float(row[tile].max()),other_row_maxabs=float(row[~tile].max()),request_start_row_maxabs=float(row[request].max()),per_request_maxabs=[float(row[i:i+512].max())for i in range(0,2048,512)],per_request_tile_edge_nonzero_rows=[int((row[i:i+512][tile[i:i+512]]!=0).sum())for i in range(0,2048,512)])
def together(call,*args):return [future.result()for future in [pools[d].submit(call,d,*args)for d in range(8)]]
try:
 report['CPU_map_contract']=validate();check(lib.deep_latent_prefill_mode_r8933(0));check(lib.deep_reference_scratch_mode_r8953(1));assert lib.deep_reference_scratch_mode_r8953(2)==1;check(lib.deep_reference_packed_batch_mode_r8996b(2));assert lib.deep_sink_broadcast_mode_r9040(2)==1;check(lib.deep_sink_broadcast_mode_r9040(1));assert lib.deep_scalar_cast_mode_r9054(2)==1
 lib.modern_deep_latent_fused_calls_r681.argtypes=[ctypes.c_int];lib.modern_deep_latent_fused_calls_r681.restype=ctypes.c_uint64
 assert [int(lib.modern_deep_latent_fused_calls_r681(d))for d in range(8)]==[0]*8;save()
 for kind in ['swa','hca128','csa4']:
  owners=together(setup,kind);assert len({v['operand_sample_SHA256']for v in owners})==8
  row=dict(kind=kind,owners=owners,measurements=[]);report['fixtures'].append(row);save()
  for which,step in enumerate(PLAN):
   arm=step['arm'];check(lib.deep_scalar_cast_mode_r9054(arm));row['measurements'].append(dict(output=which,phase=step['phase'],arm=arm,in_performance_statistics=step['phase']=='measure',owners=together(invoke,which)));save()
  pairs=[(2,b)for b in [3,4,5,6,7,8]]
  row['comparisons']={str((a,b)):together(compare,a,b)for a,b in pairs}
  row['ownrepeat_A']=all(v['maxabs']==0 for b in [5,6,8]for v in row['comparisons'][str((2,b))]);row['ownrepeat_B']=True
  row['crossarm_bitcopy_exact']=all(v['maxabs']==0 for b in [3,4,7]for v in row['comparisons'][str((2,b))]);assert row['ownrepeat_A']and row['crossarm_bitcopy_exact'];save()
  together(lambda d:state.pop(d));
 report['original_fused_call_counts']=[int(lib.modern_deep_latent_fused_calls_r681(d))for d in range(8)];assert report['original_fused_call_counts']==[0]*8
 report['completed']=True;report['status']='COMPLETED_SCALAR_CAST_COMPONENT_ONLY'
except BaseException as e:report['error']=repr(e);save();raise
finally:
 def close(d):
  torch.cuda.set_device(d);check(lib.modern_deep_latent_shutdown());return dict(device=d,thread=threading.get_native_id(),normal=True)
 report['normal_owners']=together(close);save()
 for p in pools:p.shutdown()
