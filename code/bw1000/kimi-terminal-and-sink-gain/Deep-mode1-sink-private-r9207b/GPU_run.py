import pathlib,json,hashlib,ctypes,os,time,threading
import numpy as np
import torch
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];outpath=r/'results'/('deep-mode1-sink-r9207b-'+tag+'.json');assert not outpath.exists();m=json.loads((inc/'manifest.json').read_bytes());assert sha(inc/'GPU_run.py')==m['source_SHA256']['GPU_run.py'];cp=r/'candidates/deep-mode1-sink-r9207/CPU-build.json';q=json.loads(cp.read_bytes());assert q['completed']and q['library_SHA256']==m['DSO_SHA256']and sha(q['library'])==m['DSO_SHA256'];assert all(sha(f)==h for f,h in q['reused_objects_SHA256'].items());assert all(sha(f)==v['SHA256']for f,v in q['actual_DTK_bindings'].items());torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0);lib=ctypes.CDLL(q['library']);fn=lib.modern_deep_latent_strided_r681;fn.argtypes=[ctypes.c_int]+[ctypes.c_size_t]*6+[ctypes.c_int64]*8+[ctypes.c_void_p]*9;fn.restype=ctypes.c_int;lib.modern_deep_latent_error.restype=ctypes.c_char_p;lib.modern_deep_latent_shutdown.restype=ctypes.c_int
for name in ['deep_latent_prefill_mode_r8933','deep_reference_scratch_mode_r8953','deep_reference_packed_batch_mode_r8996b','deep_mode1_sink_r9207','deep_scalar_cast_mode_r9054','deep_values_reuse_mode_r9083']:getattr(lib,name).argtypes=[ctypes.c_int]
for name in ['deep_reference_packed_batch_counts_r8996b','deep_mode1_sink_counts_r9207']:getattr(lib,name).argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
report={'completed':False,'step':tag,'scope':'ONE GPU full CSA latent component; actual B4/T1681 causal maps, synthetic RMS/RoPE operands, mode1 baseline already batched QK/PV. New defaultOFF existing sink bitcopy kernel only. Not Model quality/perf.','lengths':[413,433,458,377],'T':1681,'DSO_SHA256':m['DSO_SHA256'],'measurements':[],'normal':False}
def save():outpath.write_text(json.dumps(report,indent=2))
def check(c):
 if c:raise RuntimeError(lib.modern_deep_latent_error().decode())
def ct(name,n):
 a=(ctypes.c_uint64*n)();assert getattr(lib,name)(0,a,n)==0;return list(a)
def delta(a,b):return [v-u for u,v in zip(a,b)]
try:
 for name,v in [('deep_latent_prefill_mode_r8933',0),('deep_reference_scratch_mode_r8953',1),('deep_reference_packed_batch_mode_r8996b',1),('deep_scalar_cast_mode_r9054',1),('deep_values_reuse_mode_r9083',0)]:check(getattr(lib,name)(v))
 lengths=report['lengths'];T=sum(lengths);cr=sum(v//4 for v in lengths);si=np.full((T,128),-1,dtype=np.int64);ci=np.full((T,512),-1,dtype=np.int64);lens=np.zeros((T,2),dtype=np.int32);positions=[];qo=co=0
 for length in lengths:
  for pos in range(length):
   ns=min(pos+1,128);nc=(pos+1)//4;t=qo+pos;si[t,:ns]=qo+np.arange(pos+1-ns,pos+1);ci[t,:nc]=co+np.arange(nc);lens[t]=[ns,nc];positions.append(pos)
  qo+=length;co+=length//4
 assert co==cr and (lens[:,0]<=128).all()and(lens[:,1]<=512).all();report['metadata_SHA256']=hashlib.sha256(si.tobytes()+ci.tobytes()+lens.tobytes()).hexdigest();gen=torch.Generator(device='cuda');gen.manual_seed(920700)
 def norm(shape,rope=False):
  x=torch.randn(shape,generator=gen,device='cuda',dtype=torch.float32);x=x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()
  if rope:
   pp=torch.tensor(positions,device='cuda').float();freq=torch.pow(10000.,-torch.arange(0,64,2,device='cuda').float()/64);angle=pp[:,None]*freq[None,:];a=x[...,-64::2].clone();b=x[...,-63::2].clone();c=angle.cos();s=angle.sin()
   if len(shape)==3:c=c[:,None,:];s=s[:,None,:]
   x[...,-64::2]=a*c-b*s;x[...,-63::2]=a*s+b*c
  return x.to(torch.bfloat16)
 query=norm((T,8,512),True);swa=norm((T,512),True);comp=norm((cr,512));sink=torch.linspace(-.5,.5,8,device='cuda');args=[query,swa,comp,sink,torch.from_numpy(si).cuda(),torch.from_numpy(ci).cuda(),torch.from_numpy(lens).cuda()];clones=[x.clone()for x in args];plan=[('warm',0),('warm',1)]+[('measure',v)for v in [0,1,1,0,0,1,1,0]]+[('restore',0)];outputs=[];torch.cuda.synchronize();report['memory_start']=list(torch.cuda.mem_get_info());save()
 for phase,arm in plan:
  check(lib.deep_mode1_sink_r9207(arm));back=torch.full((query.numel()+128,),42.,device='cuda',dtype=torch.bfloat16);output=back[:query.numel()].view_as(query);before=ct('deep_mode1_sink_counts_r9207',5);beforep=ct('deep_reference_packed_batch_counts_r8996b',7);start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);stream=torch.cuda.current_stream();start.record(stream);tick=time.perf_counter();ptr=lambda x:ctypes.c_void_p(x.data_ptr());call=[0,T,8,T,cr,128,512,*query.stride(),*swa.stride(),*comp.stride(),sink.stride(0),*[ptr(x)for x in args],ptr(output),ctypes.c_void_p(stream.cuda_stream)];check(fn(*call));end.record(stream);end.synchronize();wall=time.perf_counter()-tick;gpu=start.elapsed_time(end);counts=delta(before,ct('deep_mode1_sink_counts_r9207',5));packed=delta(beforep,ct('deep_reference_packed_batch_counts_r8996b',7));assert packed[0]==1 and packed[1]==0 and packed[3]>0
  if arm:assert counts==[0,0,packed[3],packed[4],packed[4]-packed[3]],(counts,packed)
  else:assert counts==[packed[3],packed[4],0,0,0],(counts,packed)
  assert all(torch.equal(a,b)for a,b in zip(args,clones));assert torch.isfinite(output.float()).all();assert bool((back[-128:]==42).all());outputs.append(output);report['measurements'].append({'phase':phase,'arm':arm,'wall_seconds':wall,'gpu_elapsed_ms':gpu,'mode1_sink_counts':counts,'packed_counts':packed,'finite':True,'readonly':True,'canary':True,'free_total':list(torch.cuda.mem_get_info())});save();del start,end
 ref=outputs[2].float();report['numerical']=[{'arm':row['arm'],'phase':row['phase'],'maxabs':float((v.float()-ref).abs().max()),'relative_RMS':float((v.float()-ref).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-9))}for row,v in zip(report['measurements'],outputs)];report['loaded_paths']=[line for line in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(v in line for v in ['libinfiniops.so','libinfinirt.so','libinfinilm_runtime.so','_infinilm.cpython'])];report['memory_finish']=list(torch.cuda.mem_get_info());report['completed']=True
except BaseException as e:report['error']=repr(e);save();raise
finally:
 check(lib.deep_mode1_sink_r9207(0));check(lib.modern_deep_latent_shutdown());report['normal']=True;save()
