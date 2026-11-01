"""Synthetic BF16 operands at actual Deep dimensions. Component only; no model claim."""
import ctypes as C, torch, os, socket, time, json, pathlib, hashlib, statistics, traceback
S=pathlib.Path('/refactor/incoming/Deep-operator-search-r340'); OUT=pathlib.Path('/refactor/results')
local=0;rank=int(os.environ['SLURM_LOCALID'])+(8 if socket.gethostname().startswith('a02r4n05') else 0); torch.cuda.set_device(local)
start=time.monotonic(); dev=torch.device('cuda',local); torch.manual_seed(340+rank)
q=dict(rank=rank,local=local,host=socket.gethostname(),step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],completed=False,scope='Same-shape synthetic component. Custom unfused baseline, NOT production Runtime benchmark. No model throughput claim.',rows=[])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q['source_SHA256']={p.name:sha(p) for p in S.iterdir() if p.is_file()}; q['torch']=torch.__version__;q['gpu']=torch.cuda.get_device_name(local)
lib=C.CDLL(str(S/'lib340.so')); fn=lib.run340; fn.restype=C.c_int
fn.argtypes=[C.c_void_p]*4+[C.c_int]*5+[C.c_float,C.c_void_p]
stream=torch.cuda.current_stream().cuda_stream
def save():
 p=OUT/f'Deep340-{q["step"]}-rank{rank}.json'; p.write_text(json.dumps(q,indent=2)+'\n')
try:
 # Fixed nonoverlapping distribution: each rank owns a dimension/T family and tests all meaningful B/fusion arms.
 families=[(1,128,1),(1,512,1),(16,128,1),(16,512,1),(256,128,1),(256,512,1),(512,128,1),(512,512,1),(1,4096,0),(8,4096,0),(32,4096,0),(128,4096,0),(256,4096,0),(512,4096,0),(2048,128,1),(2048,512,1)]
 T,D,rotate=families[rank];q['shape']=[T,D];q['rotate']=rotate
 x=torch.randn(T,D,device=dev,dtype=torch.bfloat16);w=(1+torch.randn(D,device=dev)*.03).bfloat16();y=torch.empty_like(x)
 angles=torch.arange(T,device=dev).float()[:,None]*torch.exp(-torch.arange(32,device=dev).float()[None,:]/32*9.21034037)
 table=torch.cat((angles.cos(),angles.sin()),dim=1).contiguous()
 ref=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*w.float()).bfloat16()
 if rotate:
  a=ref[:,-64:-32].float();b=ref[:,-32:].float();ref=torch.cat((ref[:,:-64],(a*table[:,:32]-b*table[:,32:]).bfloat16(),(b*table[:,:32]+a*table[:,32:]).bfloat16()),dim=-1)
 def call(B,F):
  rc=fn(x.data_ptr(),w.data_ptr(),y.data_ptr(),table.data_ptr(),T,D,B,F,rotate,1e-6,stream);assert rc==0,rc
 def measure(B,F):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();t=time.perf_counter();a.record()
  for _ in range(100):call(B,F)
  b.record();b.synchronize();return dict(wall_ms=(time.perf_counter()-t)*10,event_ms=a.elapsed_time(b)/100,launches_per_call=1 if F or not rotate else 2,scope='100 ctypes submissions including Python host gaps; events on current stream, no allocations/reference in interval')
 for B in [64,128,256,512,1024]:
  for F in ([0,1] if rotate else [0]):
   for _ in range(20):call(B,F)
   torch.cuda.synchronize();d=(y.float()-ref.float());num=dict(finite=bool(torch.isfinite(y).all()),max_abs=float(d.abs().max()),nrmse=float(d.square().mean().sqrt()/(ref.float().square().mean().sqrt()+1e-12)))
   assert num['finite'] and num['nrmse']<.01,num
   pairs=[]
   for rep in range(5):
    aa=measure(256,0);bb=measure(B,F);cc=measure(B,F);dd=measure(256,0);pairs.append(dict(A1=aa,B1=bb,B2=cc,A2=dd,wall_ratio=(bb['wall_ms']+cc['wall_ms'])/(aa['wall_ms']+dd['wall_ms']),baseline_drift=dd['wall_ms']/aa['wall_ms']-1))
   q['rows'].append(dict(B=B,fused=F,numerics=num,pairs=pairs,median_wall_ratio=statistics.median(p['wall_ratio']for p in pairs)));save()
 q['completed']=True
except BaseException as e:q['error']=repr(e);q['traceback']=traceback.format_exc();raise
finally:
 torch.cuda.synchronize();q['elapsed_s']=time.monotonic()-start;save();print(json.dumps({k:q.get(k)for k in ['rank','host','step','completed','elapsed_s','error']}),flush=True)
