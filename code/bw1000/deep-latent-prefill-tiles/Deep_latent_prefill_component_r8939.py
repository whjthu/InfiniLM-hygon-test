"""Finite synthetic attention operands; component only, never actualModel profile."""
import pathlib,json,hashlib,ctypes,os,time,threading,concurrent.futures
import numpy as np
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=r/'results'/('deep-latent-prefill-component-r8939-'+tag+'.json');assert not path.exists();m=json.loads((r/'incoming/Deep-latent-prefill-component-manifest-r8939.json').read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();bp=r/m['cpu_receipt_path'];assert sha(bp)==m['existing_remote_SHA256'][m['cpu_receipt_path']];cpu=json.loads(bp.read_text());assert cpu['completed']and cpu['CPU_only']and cpu['libraries']['full']['SHA256']==m['DSO_SHA256'];artifact=cpu['libraries']['full']['path'];assert sha(artifact)==m['DSO_SHA256']
for n,h in cpu['link']['full']['objects'].items():assert sha(n)==h
for n,v in cpu['actual_DTK_bindings'].items():assert sha(n)==v['SHA256']
import torch
torch.set_num_threads(1);assert torch.cuda.device_count()==8;lib=ctypes.CDLL(artifact);fn=lib.modern_deep_latent_strided_r681;fn.argtypes=[ctypes.c_int]+[ctypes.c_size_t]*6+[ctypes.c_int64]*8+[ctypes.c_void_p]*9;fn.restype=ctypes.c_int;lib.modern_deep_latent_error.restype=ctypes.c_char_p;lib.modern_deep_latent_shutdown.restype=ctypes.c_int;lib.deep_latent_prefill_mode_r8933.argtypes=[ctypes.c_int];lib.deep_latent_prefill_counts_r8933.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
report=dict(hardware='BW1000',level='SYNTHETIC_ATTENTION_COMPONENT',status='RUNNING',completed=False,step=tag,DSO_SHA256=m['DSO_SHA256'],scope='Synthetic q/KV/sink with causalSWA+CSA selection, T2048H8D512; samecurrentSDK/fullreference vs optinquerytiles32. NoModelweights/load/CCL; noModelgain orkernelactiveclaim.',quality=[],timed=[],normal_owners=[])
def save():
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
def check(code):
 if code:raise RuntimeError(lib.modern_deep_latent_error().decode())
state={};pools=[concurrent.futures.ThreadPoolExecutor(max_workers=1)for _ in range(8)]
def setup(d,T=2048,kind='csa'):
 torch.cuda.set_device(d);torch.manual_seed(8939+d);H=8;sw=128;cw=512 if kind=='csa'else 1;sr=2048;cr=512
 q=(torch.randn(T,H,512,device='cuda')*.3).to(torch.bfloat16);swa=(torch.randn(sr,512,device='cuda')*.3).to(torch.bfloat16);comp=(torch.randn(cr,512,device='cuda')*.3).to(torch.bfloat16);sink=torch.randn(H,device='cuda',dtype=torch.float32)*.1
 si=np.full((T,sw),-1,np.int64);ci=np.full((T,cw),-1,np.int64);lens=np.zeros((T,2),np.int32)
 for t in range(T):
  req=(t//512)%4;pos=t%512;ns=min(pos+1,128);nc=(pos+1)//4 if kind=='csa'else 0;lens[t]=[ns,nc];si[t,:ns]=req*512+np.arange(pos+1-ns,pos+1);ci[t,:nc]=req*128+np.arange(nc)
 outs=[torch.empty_like(q)for _ in range(2)];a=(q,swa,comp,sink,torch.from_numpy(si).cuda(),torch.from_numpy(ci).cuda(),torch.from_numpy(lens).cuda());state[d]=dict(args=a,outs=outs,T=T,H=H,sw=sw,cw=cw,sr=sr,cr=cr,thread=threading.get_native_id());return d

def counts(d):
 a=(ctypes.c_uint64*3)();check(lib.deep_latent_prefill_counts_r8933(d,a,3));return list(a)
def invoke(d,which):
 z=state[d];assert threading.get_native_id()==z['thread'];torch.cuda.set_device(d);q,s,c,sink,si,ci,lens=z['args'];ptr=lambda x:ctypes.c_void_p(x.data_ptr());args=[d,z['T'],z['H'],z['sr'],z['cr'],z['sw'],z['cw'],*q.stride(),*s.stride(),*c.stride(),sink.stride(0),*[ptr(x)for x in z['args']],ptr(z['outs'][which]),ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)];before=counts(d);torch.cuda.synchronize();tick=time.perf_counter();check(fn(*args));elapsed=(time.perf_counter()-tick)*1000;after=counts(d);delta=[b-a for a,b in zip(before,after)];t=z['T'];eligible=which==1 and 64<t<=4096 and(t%32==0 or t%32>=8);assert delta==([1,t,(t+31)//32]if eligible else[0,0,0]);return dict(device=d,thread=z['thread'],stream=int(torch.cuda.current_stream().cuda_stream),ms=elapsed,counter_delta=[b-a for a,b in zip(before,after)])
def together(call,*args):return [f.result()for f in [pools[d].submit(call,d,*args)for d in range(8)]]
def compare(d):
 a,b=state[d]['outs'];assert torch.isfinite(a.float()).all()and torch.isfinite(b.float()).all();x=a.float();delta=b.float()-x;den=x.square().mean().sqrt().clamp_min(1e-9);return dict(device=d,maxabs=float(delta.abs().max()),nrmse=float(delta.square().mean().sqrt()/den),p99abs=float(torch.quantile(delta.abs().flatten(),.99)),bit_equal=bool(torch.equal(a,b)))
try:
 save();together(setup);check(lib.deep_latent_prefill_mode_r8933(0));control=together(invoke,0);check(lib.deep_latent_prefill_mode_r8933(1));candidate=together(invoke,1);report['quality'].append(dict(label='T2048_causalCSA',control=control,candidate=candidate,error=together(compare)));save()
 # Own-repeat is deterministic, compared remotely without saving output tensors.
 def freeze(d):state[d]['frozen']=state[d]['outs'][1].clone()
 together(freeze);repeat=together(invoke,1)
 def ownrepeat(d):return bool(torch.equal(state[d]['frozen'],state[d]['outs'][1]))
 report['ownrepeat']=together(ownrepeat);assert all(report['ownrepeat']);save()
 for arm in [0,1,1,0,1,0,0,1]:
  check(lib.deep_latent_prefill_mode_r8933(arm));report['timed'].append(dict(arm=arm,owners=together(invoke,arm)));save()
 for T in [65,72,95]:
  together(setup,T,'swa');check(lib.deep_latent_prefill_mode_r8933(0));a=together(invoke,0);check(lib.deep_latent_prefill_mode_r8933(1));b=together(invoke,1);report['quality'].append(dict(label='SWA_tail_'+str(T),control=a,candidate=b,error=together(compare)));save()
 def cleanup(d):
  check(lib.modern_deep_latent_shutdown());return dict(device=d,thread=threading.get_native_id(),shutdown_returncode=0)
 report['normal_owners']=together(cleanup);report.update(completed=True,status='QUALIFIED_COMPONENT_PENDING_AUDIT');save()
except BaseException as e:report.update(error=repr(e),status='FAILED');save();raise
finally:
 for pool in pools:pool.shutdown(wait=True)