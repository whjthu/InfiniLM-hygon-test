"""T16 actual checkpoint TP8 shard component; seeded latent/routes, not captured traffic."""
import os,pathlib,json,hashlib,ctypes,time,statistics,contextlib,gc
import numpy as np
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/shared/Kimi-K3');L=1;BANK=0
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def worker(rank,manifest,output):
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(rank)
 m=json.loads(pathlib.Path(manifest).read_text());assert sha(m['library'])==m['library_SHA256'];lib=ctypes.CDLL(m['library'])
 original=lib.modern_kimi_grouped_r4620;original.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*15+[ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p]
 candidate=lib.modern_kimi_grouped_events_r8661;candidate.argtypes=original.argtypes+[ctypes.c_void_p,ctypes.c_void_p]
 out=pathlib.Path(str(output)+f'-rank{rank}.json');d={'rank':rank,'normal_CUDA_cleanup':False,'scope':'RealcheckpointTP8localweights; T16 seededBF16latent/routing preliminarycomponent, no captured currenttraffic or Modeltokens'}
 def save():
  q=out.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(out)
 def check(code):
  if code:raise RuntimeError('nativeCUDAstatus:'+str(code))
 def ptr(t):return ctypes.c_void_p(t.data_ptr())
 def bits(t):return t.detach().cpu().view(torch.uint16).numpy().copy()
 def fp(v):return (v.astype(np.uint32)<<16).view(np.float32)
 save()
 try:
  cfg=json.loads((model/'config.json').read_text())['text_config'];assert cfg['hidden_size']==7168 and cfg['routed_expert_hidden_size']==3584 and cfg['moe_intermediate_size']//8==384
  mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];prefix=f'language_model.model.layers.{L}.block_sparse_moe.experts.';E=896
  host={'w13':torch.empty((E,768,1792),dtype=torch.uint8),'s13':torch.empty((E,768,112),dtype=torch.uint8),'w2':torch.empty((E,3584,192),dtype=torch.uint8),'s2':torch.empty((E,3584,12),dtype=torch.uint8)}
  tick=time.perf_counter();files={}
  with contextlib.ExitStack()as stack:
   def get(expert,name,axis,width):
    key=prefix+str(expert+BANK)+'.'+name;fn=mapping[key]
    if fn not in files:files[fn]=stack.enter_context(safe_open(model/fn,framework='pt',device='cpu'))
    view=files[fn].get_slice(key);sl=[slice(None)]*len(view.get_shape());sl[axis]=slice(rank*width,(rank+1)*width);v=view[tuple(sl)].contiguous();assert v.dtype==torch.uint8;return v
   for e in range(E):
    host['w13'][e,:384].copy_(get(e,'w1.weight_packed',0,384));host['w13'][e,384:].copy_(get(e,'w3.weight_packed',0,384));host['s13'][e,:384].copy_(get(e,'w1.weight_scale',0,384));host['s13'][e,384:].copy_(get(e,'w3.weight_scale',0,384));host['w2'][e].copy_(get(e,'w2.weight_packed',1,192));host['s2'][e].copy_(get(e,'w2.weight_scale',1,12))
  d['partial_weights_load_seconds']=time.perf_counter()-tick;d['selected_checkpoint_tensors']=E*6;d['host_weights_sha256']={k:hashlib.sha256(v.numpy().tobytes()).hexdigest()for k,v in host.items()};d['config_sha256']=sha(model/'config.json');d['index_sha256']=sha(model/'model.safetensors.index.json');weights={k:v.cuda()for k,v in host.items()};torch.cuda.synchronize();save()

  T=16;rng=np.random.default_rng(4391+T*10+1);x_cpu=torch.from_numpy(rng.standard_normal((T,3584),dtype=np.float32)*.1).to(torch.bfloat16)
  ids_cpu=np.stack([rng.choice(E,16,replace=False).astype(np.int32)for _ in range(T)]);rw=rng.uniform(.01,1,(T,16)).astype(np.float32);rw/=rw.sum(-1,keepdims=True)
  d['fixture_SHA256']={'input':hashlib.sha256(x_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),'ids':hashlib.sha256(ids_cpu.tobytes()).hexdigest(),'routes':hashlib.sha256(rw.tobytes()).hexdigest()}
  x=x_cpu.cuda();ids=torch.from_numpy(ids_cpu).cuda();routes=torch.from_numpy(rw).cuda();activation=torch.empty((T,16,384),device='cuda',dtype=torch.bfloat16);y=torch.empty_like(x)
  workspace=[torch.empty((E,),device=rank,dtype=torch.int32),torch.empty((E,16*T),device=rank,dtype=torch.int32),torch.empty((16*T,),device=rank,dtype=torch.int32),torch.empty((16*T,),device=rank,dtype=torch.int32),torch.empty((1,),device=rank,dtype=torch.int32)];partial=torch.empty((T*16,3584),device=rank,dtype=torch.float32)
  stream=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream);wp=[ptr(weights[k])for k in ['w13','s13','w2','s2']]
  events=[torch.cuda.Event(enable_timing=True)for _ in range(4)]
  for ev in events:ev.record()
  torch.cuda.synchronize();begin,end,wbegin,wend=events
  args=[rank,4,ptr(y),ptr(activation),ptr(x),ptr(ids),ptr(routes),*wp,*[ptr(z)for z in workspace],ptr(partial),E,T,stream]
  def invoke(on=None):
   if on is None:check(original(*args))
   else:
    os.environ['INFINILM_KIMI_W13_WAVE_ACT_R8660']='1'if on else'0'
    check(candidate(*args,ctypes.c_void_p(wbegin.cuda_event),ctypes.c_void_p(wend.cuda_event)))
  invoke();torch.cuda.synchronize();goldA,goldY=bits(activation),bits(y)
  invoke(False);torch.cuda.synchronize();baseA,baseY=bits(activation),bits(y);d['defaultOFF_faithful_original']={'activation_exact':bool(np.array_equal(baseA,goldA)),'output_exact':bool(np.array_equal(baseY,goldY))};save()
  assert all(d['defaultOFF_faithful_original'].values()),'newdefaultOFF didnotreproduceactualimmutableprovider'
  invoke(True);torch.cuda.synchronize();testA,testY=bits(activation),bits(y)
  d['candidate_vs_original']={}
  for key,a,b in [('activation',testA,goldA),('output',testY,goldY)]:
   aa,bb=fp(a),fp(b);assert np.isfinite(aa).all()and np.isfinite(bb).all();d['candidate_vs_original'][key]={'changed':int(np.count_nonzero(a!=b)),'elements':a.size,'maxabs':float(np.abs(aa-bb).max()),'nrmse':float(np.sqrt(np.mean((aa.astype(np.float64)-bb)**2))/max(1e-30,np.sqrt(np.mean(bb.astype(np.float64)**2))))}
  hist=workspace[0].cpu().numpy();assert np.array_equal(hist,np.bincount(ids_cpu.reshape(-1),minlength=E));d['actual_route_count']=int(hist.sum());d['unique_experts']=int(np.count_nonzero(hist));d['samples']=[]
  for _ in range(3):
   invoke(False);invoke(True)
  torch.cuda.synchronize()
  for pair in range(8):
   for on in ([False,True]if pair%2==0 else[True,False]):
    torch.cuda.synchronize();start=time.perf_counter();begin.record();invoke(on);end.record();end.synchronize();wall=(time.perf_counter()-start)*1000
    d['samples'].append({'pair':pair,'candidate':on,'full_provider_GPU_event_ms':begin.elapsed_time(end),'W13_only_GPU_event_ms':wbegin.elapsed_time(wend),'full_eager_host_ms':wall})
  # Both outputs rechecked after timed samples, outside event spans.
  invoke(False);torch.cuda.synchronize();assert np.array_equal(bits(activation),baseA)and np.array_equal(bits(y),baseY)
  invoke(True);torch.cuda.synchronize();assert np.array_equal(bits(activation),testA)and np.array_equal(bits(y),testY)
  for key in ['full_provider_GPU_event_ms','W13_only_GPU_event_ms','full_eager_host_ms']:
   ratios=[]
   for pair in range(8):
    vals={v['candidate']:v[key]for v in d['samples']if v['pair']==pair};ratios.append(vals[False]/vals[True])
   d.setdefault('paired_ratios',{})[key]=ratios
  d['performance_scope']='Fullprovider includes originalroutepack/jobs+W13+unchangedW2; W13 events only around W13 afterbuildJobs. Samecallerstream/events botharms. Per-rank independentTP8shard component, noTPcollective/Model throughput. 3warm+8balancedAB/BApairs.'
  torch.cuda.synchronize();del weights,host,x,ids,routes,activation,y,workspace,partial,begin,end,wbegin,wend,events;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();d['normal_CUDA_cleanup']=True;save()
 except BaseException as e:d['error']=repr(e);save();raise
if __name__=='__main__':
 import sys
 manifest,output=sys.argv[1:];mp.spawn(worker,args=(manifest,output),nprocs=8,join=True)
 rows=[json.loads(pathlib.Path(output+'-rank'+str(i)+'.json').read_text())for i in range(8)];assert all(v['normal_CUDA_cleanup']and'error'not in v for v in rows)
 pathlib.Path(output).write_text(json.dumps({'all8normal':True,'scope':'T16 actualcheckpoint seededfixture component only; noModel tokens/TPS; noRuntime/CCL instantiated','workers':rows},indent=2))
