"""Three finite shapes, one actual layer38 load; no Model TPS claim."""
import pathlib,json,hashlib,os,ctypes,time,statistics,contextlib
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');base=r/'candidates/glm-matrix-proof-r8509';model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3')
SHA='058141002f2df00aeeb666d34ffcfc8692889ee7d9bc41d5c60fc583ebb1a9dd'
def worker(rank):
 torch.set_num_threads(1);torch.cuda.set_device(rank)
 cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=32;os.sched_setaffinity(0,cpus[4*rank:4*rank+4])
 libpath=base/'libglm_matrix_r8514.so';assert hashlib.sha256(libpath.read_bytes()).hexdigest()==SHA
 lib=ctypes.CDLL(str(libpath));V=ctypes.c_void_p;Z=ctypes.c_size_t
 for name in ['glm_matrix_gate_r8508','glm_routed_gate_r19']:getattr(lib,name).argtypes=[V]*9+[Z,Z,V]
 for name in ['glm_matrix_down_r8508','glm_routed_down_r19']:getattr(lib,name).argtypes=[V]*6+[Z,Z,V]
 out=r/'results'/f'glm-matrix-finite-r8518-{os.uname().nodename}-rank{rank}.json';assert not out.exists()
 d={'completed':False,'rank':rank,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'affinity':sorted(os.sched_getaffinity(0)),'layer':38,'offset':rank*256,'library_SHA256':SHA,'cases':[],'scope':'Actual layer38 weights, synthetic BF16 activation and frozen constructed sparse routeIDs. Isolated projections, no router/attention/CCL/shared-expert/ModelTPS; matrix M16 uses only1 useful row.'}
 def save():out.write_text(json.dumps(d,indent=2))
 def ptr(x):return V(x.data_ptr())
 banks=[];scales=[];held=[]
 try:
  mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];offset=rank*256;t=time.perf_counter()
  with contextlib.ExitStack()as stack:
   handles={}
   def load(k):
    name=mapping[k]
    if name not in handles:handles[name]=stack.enter_context(safe_open(str(model/name),framework='pt',device='cpu'))
    return handles[name].get_tensor(k)
   for proj in ['gate_proj','up_proj','down_proj']:
    ww=[];ss=[]
    for e in range(256):
     prefix=f'model.layers.38.mlp.experts.{e}.{proj}';w=load(prefix+'.weight').view(torch.uint8);v=w[:,offset:offset+256]if proj=='down_proj'else w[offset:offset+256,:]
     ww.append(v.contiguous().clone());ss.append(load(prefix+'.weight_scale_inv').clone())
    banks.append(torch.stack(ww).cuda());scales.append(torch.stack(ss).float().cuda());del ww,ss
  torch.cuda.synchronize();d['weight_load_seconds']=time.perf_counter()-t;d['weight_load_count']=1;d['free_after_load']=torch.cuda.mem_get_info()[0];assert d['free_after_load']>=6*1024**3;save()
  def quant(x):
   q=x.float().view(x.shape[0],-1,128);s=q.abs().amax(-1).clamp_min(1e-10)/448.;return (q/s[:,:,None]).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).reshape(x.shape).contiguous(),s.contiguous()
  def oracle(packed,xs,ids,gate,channels):
   # Float64 CPU dot is independent diagnostic; GPU and communication stay F32.
   xx=packed.cpu().view(torch.float8_e4m3fn).double();sx=xs.cpu().double();ii=ids.cpu().tolist();answer=[]
   for route,ch in [(0,channels[0]),(len(ii)//2,channels[1]),(len(ii)-1,channels[2])]:
    token=route//8 if gate else route;e=ii[route];which=0 if gate else 2;n=ch
    if gate and ch>=256:which=1;n=ch-256
    weight=banks[which][e,n].cpu().view(torch.float8_e4m3fn).double().reshape(-1,128);a=xx[token].reshape(-1,128)
    sw=scales[which][e].cpu().double();total=0.
    for b in range(a.shape[0]):
     sc=sw[(offset+n)//128,b]if gate else sw[n//128,offset//128+b]
     total+=float((a[b]*weight[b]).sum())*float(sx[token,b])*float(sc)
    answer.append({'route':route,'channel':ch,'FP64_CPU_reference':total})
   return answer
  for tokens,U in [(1,8),(4,27),(8,53)]:
   torch.manual_seed(851800+rank*13+tokens);routes=tokens*8
   unique=[(j*37+rank*17)%256 for j in range(U)];rr=unique+[unique[j%U]for j in range(routes-U)];ids=torch.tensor(rr,dtype=torch.int64,device='cuda');assert max(rr)<256
   x=torch.randn((tokens,6144),device='cuda',dtype=torch.bfloat16)*.1;y=torch.randn((routes,256),device='cuda',dtype=torch.bfloat16)*.1
   packed,xs=quant(x);dp,ds=quant(y);row={'tokens':tokens,'routes':routes,'U':U,'max_rows_per_expert':max(rr.count(v)for v in set(rr)),'useful_M':1,'matrix_M':16,'executed_to_useful_FLOPs_ratio':16,'routeIDs_SHA256':hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),'projections':[]}
   for gate in [True,False]:
    z,s=(packed,xs)if gate else(dp,ds)
    # Flat contiguous output with guard elements before/after the full region.
    a=torch.full((routes*(256 if gate else 6144)+2,),123.,dtype=torch.bfloat16,device='cuda');bb=torch.full_like(a,123.)if gate else None
    aa=a[1:-1].view(routes,256 if gate else 6144);bu=bb[1:-1].view(routes,256)if gate else None
    def call(mode):
     stream=V(torch.cuda.current_stream().cuda_stream)
     if gate:
      f=getattr(lib,'glm_matrix_gate_r8508'if mode else'glm_routed_gate_r19');ret=f(ptr(aa),ptr(bu),ptr(z),ptr(s),ptr(ids),ptr(banks[0]),ptr(scales[0]),ptr(banks[1]),ptr(scales[1]),routes,offset,stream)
     else:
      f=getattr(lib,'glm_matrix_down_r8508'if mode else'glm_routed_down_r19');ret=f(ptr(aa),ptr(z),ptr(s),ptr(ids),ptr(banks[2]),ptr(scales[2]),routes,offset,stream)
     assert ret==0,('kernel_launch',ret)
    refs=oracle(z,s,ids,gate,[0,255,511]if gate else[0,3071,6143]);arms=[];baseline=None
    for mode in [0,1,1,0]:
     for _ in range(3):call(mode)
     torch.cuda.synchronize();samples=[]
     for _ in range(3):
      st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);start=time.perf_counter();st.record()
      for j in range(3):call(mode)
      en.record();en.synchronize();samples.append({'event_ms':st.elapsed_time(en)/3,'wall_ms':(time.perf_counter()-start)*1000/3})
     got=torch.cat([aa,bu],dim=1)if gate else aa.clone();assert torch.isfinite(got).all();assert a[0]==123 and a[-1]==123
     if gate:assert bb[0]==123 and bb[-1]==123
     if baseline is None:baseline=got.clone()
     if mode==0:assert torch.equal(got,baseline),'scalar_restore'
     oracle_rows=[]
     for ref in refs:
      v=float(got[ref['route'],ref['channel']]);err=abs(v-ref['FP64_CPU_reference']);oracle_rows.append(dict(ref,observed=v,abs_error=err));assert err<=.03125+.02*abs(ref['FP64_CPU_reference']),oracle_rows
     diff=(got.float()-baseline.float());arms.append({'mode':mode,'samples':samples,'event_median_ms':statistics.median(v['event_ms']for v in samples),'wall_median_ms':statistics.median(v['wall_ms']for v in samples),'finite':True,'canary':True,'max_abs_vs_scalar':float(diff.abs().max()),'relativeL2_vs_scalar':float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(baseline.float()).clamp_min(1e-30)),'own_arm_CPU_oracle':oracle_rows})
    row['projections'].append({'gate_up':gate,'arms':arms,'scalar_restore_exact':True});del aa,bu,a,bb,baseline,got
   d['cases'].append(row);save();del x,y,packed,xs,dp,ds,ids
  d['normal_drain_before_release']=True
 finally:
  # All device operands survive until this drain on either normal or exceptional return.
  torch.cuda.synchronize();banks.clear();scales.clear();held.clear();torch.cuda.empty_cache()
 torch.ones(1,device='cuda').add_(1);torch.cuda.synchronize();d.update(completed=True,normal_Torch_close=True);save()
if __name__=='__main__':
 assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
 mp.spawn(worker,nprocs=8,join=True)
