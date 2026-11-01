import os,json,ctypes,time,pathlib,hashlib,socket
import torch
torch.set_num_threads(1)
dev=int(os.environ['LOCAL_GPU']);arm=int(os.environ['ARM']);seed=int(os.environ['SEED']);torch.cuda.set_device(dev)
D=pathlib.Path('/refactor/incoming/Kimi-operator-search-r339');lib=ctypes.CDLL(str(D/'search.so'));lib.search_run.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p]
torch.manual_seed(seed);device='cuda:'+str(dev)
def rand(shape,scale=.2,dtype=torch.bfloat16):return (torch.randn(shape,device=device)*scale).to(dtype)
initial=rand((3,6,128,128));state=initial.clone();q,k,v,g=[rand((6,128)) for _ in range(4)];b=rand((6,));a=rand((6,),dtype=torch.float32);bias=rand((6,128),dtype=torch.float32);out=torch.empty_like(q);snap=torch.empty_like(state);off=torch.tensor([0,1],device=device,dtype=torch.int32);read=torch.tensor([1],device=device,dtype=torch.int32);write=torch.tensor([2],device=device,dtype=torch.int32);tmp=torch.empty(2304,device=device,dtype=torch.float32)
args=[state,q,k,v,g,b,a,bias,out,snap,off,read,write,tmp];ptr=(ctypes.c_void_p*14)(*[x.data_ptr() for x in args]);stream=torch.cuda.current_stream().cuda_stream
def run(which):
 rc=lib.search_run(which,ptr,stream)
 assert rc==0,('CUDA rc',rc)
def reset():state.copy_(initial);torch.cuda.synchronize()
reset();run(0);torch.cuda.synchronize();gold=out.clone();goldstate=state.clone()
reset();run(arm);torch.cuda.synchronize()
err={}
for name,x,y in [('output',out,gold),('state',state,goldstate)]:
 assert torch.isfinite(x).all() and torch.isfinite(y).all()
 diff=x.float()-y.float();err[name]={'max_abs':diff.abs().max().item(),'rms':diff.square().mean().sqrt().item(),'baseline_rms':y.float().square().mean().sqrt().item()}
 assert torch.allclose(x.float(),y.float(),atol=.004,rtol=.04),(name,err[name])
assert torch.equal(state[:2],initial[:2]),'read and canary slots changed'
# CPU F32 independently checks the mathematical equation; output BF16/state BF16 rounding is allowed.
Q=q.float().cpu();K=k.float().cpu();V=v.float().cpu();G=g.float().cpu();S=initial[1].float().cpu();A=a.cpu();Bias=bias.cpu();Beta=b.float().cpu()
Q=Q*torch.rsqrt(Q.square().sum(-1,keepdim=True)+1e-6)/128**.5;K=K*torch.rsqrt(K.square().sum(-1,keepdim=True)+1e-6)
old=S*torch.exp(-5*torch.sigmoid(torch.exp(A[:,None])*(G+Bias)))[:,None,:];delta=(V-(old*K[:,None,:]).sum(-1))*torch.sigmoid(Beta[:,None]);refout=((old*Q[:,None,:]).sum(-1)+delta*(K*Q).sum(-1)[:,None]).bfloat16();refstate=(old+delta[:,:,None]*K[:,None,:]).bfloat16()
assert torch.allclose(out.cpu().float(),refout.float(),atol=.004,rtol=.04);assert torch.allclose(state[2].cpu().float(),refstate.float(),atol=.004,rtol=.04)
rounds=[];start=time.monotonic()
for repeat in range(8):
 for mode in ([0,arm,arm,0] if repeat%2==0 else [arm,0,0,arm]):
  reset()
  for _ in range(20):run(mode)
  torch.cuda.synchronize();s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);t0=time.perf_counter();s.record()
  for _ in range(100):run(mode)
  e.record();e.synchronize();wall=(time.perf_counter()-t0)*10;event=s.elapsed_time(e)/100
  rounds.append({'round':repeat,'arm':mode,'wall_ms':wall,'GPUevent_ms':event})
  assert time.monotonic()-start<540
result={'host':socket.gethostname(),'dev':dev,'arm':arm,'seed':seed,'device':torch.cuda.get_device_name(dev),'shape':{'C':1,'T':1,'H':6,'D':128,'pool':3,'read':1,'write':2},'input_kind':'synthetic BF16; current Math0 C1 geometry; not captured activation','numerics':err,'CPU_F32_reference':True,'finite':True,'canary_read_unchanged':True,'rounds':rounds,'elapsed_measure_s':time.monotonic()-start,'scope':'standalone indexed KDA launch including original pool snapshot; no Delta/CCL/model gain claim','source_commit':os.environ['SOURCE_COMMIT'],'source_manifest_SHA256':hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),'binary_SHA256':hashlib.sha256((D/'search.so').read_bytes()).hexdigest()}
target=pathlib.Path('/refactor/results')/('Kimi339-'+socket.gethostname()+'-gpu'+str(dev)+'.json');target.write_text(json.dumps(result,indent=2));print(json.dumps({'result':str(target),'arm':arm,'elapsed_s':result['elapsed_measure_s']}))
