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
 cold=lib.kimi_bf16_cold_run_r8666;cold.argtypes=[ctypes.c_void_p,ctypes.c_size_t]+[ctypes.c_void_p]*16
 lib.kimi_bf16_cold_create_r8666.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];lib.kimi_bf16_cold_create_r8666.restype=ctypes.c_void_p
 lib.kimi_bf16_cold_destroy_r8666.argtypes=[ctypes.c_void_p];lib.kimi_bf16_cold_error_r8666.restype=ctypes.c_char_p
 out=pathlib.Path(str(output)+f'-rank{rank}.json');d={'rank':rank,'normal_CUDA_cleanup':False,'scope':'RealcheckpointTP8localweights; T16 seededBF16latent/routing preliminarycomponent, no captured currenttraffic or Modeltokens'}
 def save():
  q=out.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(out)
 def check(code):
  if code:raise RuntimeError('nativeCUDAstatus:'+str(code)+':'+str(lib.kimi_bf16_cold_error_r8666()))
 def ptr(t):return ctypes.c_void_p(t.data_ptr())
 def bits(t):return t.detach().cpu().view(torch.uint16).numpy().copy()
 def fp(v):return (v.astype(np.uint32)<<16).view(np.float32)
 save()
 ctx=None
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


  # One fixed scratchset reused across bothfixtures/allcalls, never persistentweights.
  vendor=torch.empty((64*1024*1024,),device=rank,dtype=torch.uint8);bank=torch.empty((32,768,3584),device=rank,dtype=torch.bfloat16);packX=torch.empty((32,16,3584),device=rank,dtype=torch.bfloat16);gemm=torch.empty((32,16,768),device=rank,dtype=torch.float32);active=torch.empty((256,),device=rank,dtype=torch.int32);count=torch.empty((1,),device=rank,dtype=torch.int32)
  d['explicit_new_scratch_bytes']=sum(v.numel()*v.element_size()for v in [vendor,bank,packX,gemm,active,count]);assert d['explicit_new_scratch_bytes']<256*1024**2
  stream=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream);free_before=torch.cuda.mem_get_info()[0];ctx=lib.kimi_bf16_cold_create_r8666(rank,stream,ptr(vendor),vendor.numel());assert ctx,lib.kimi_bf16_cold_error_r8666();d['vendor_create_free_delta_bytes']=free_before-torch.cuda.mem_get_info()[0];d['fixtures']=[];save()
  refs=m['captured_fixture_refs'][str(rank)];actual={}
  for point,z in refs.items():
   f=pathlib.Path(z['file']);assert sha(f)==z['SHA256'];actual[point]=np.fromfile(f,{'I32':np.int32,'F32':np.float32,'BF16':np.uint16}[z['dtype']]).reshape(z['shape'])
  T=16;rng=np.random.default_rng(4391+16*10+1);seedX=torch.from_numpy(rng.standard_normal((16,3584),dtype=np.float32)*.1).to(torch.bfloat16);seedIDs=np.stack([rng.choice(896,16,replace=False).astype(np.int32)for _ in range(16)]);seedRW=rng.uniform(.01,1,(16,16)).astype(np.float32);seedRW/=seedRW.sum(-1,keepdims=True)
  fixtures=[('captured8_tiled_to16',torch.from_numpy(np.tile(actual['routed_input'],(2,1)).copy()).view(torch.bfloat16),np.tile(actual['routing_ids'],(2,1)).copy(),np.tile(actual['routing_weights'],(2,1)).copy()),('seed229',seedX,seedIDs,seedRW)]
  for label,x_cpu,ids_cpu,rw in fixtures:
   assert list(x_cpu.shape)==[16,3584]and list(ids_cpu.shape)==[16,16]and list(rw.shape)==[16,16];assert ((ids_cpu>=0)&(ids_cpu<896)).all()and all(len(np.unique(v))==16 for v in ids_cpu)
   row={'label':label,'T':16,'active_experts':len(np.unique(ids_cpu)),'chunk_count':(len(np.unique(ids_cpu))+31)//32,'per_token_topk16_unique_checked':True,'fixture_SHA256':{'input':hashlib.sha256(x_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),'ids':hashlib.sha256(ids_cpu.tobytes()).hexdigest(),'routes':hashlib.sha256(rw.tobytes()).hexdigest()}}
   assert row['active_experts']==(21 if label=='captured8_tiled_to16'else 229)
   x=x_cpu.cuda();ids=torch.from_numpy(ids_cpu).cuda();routeweights=torch.from_numpy(rw).cuda();activation=torch.empty((16,16,384),device=rank,dtype=torch.bfloat16);y=torch.empty_like(x)
   workspace=[torch.empty((896,),device=rank,dtype=torch.int32),torch.empty((896,256),device=rank,dtype=torch.int32),torch.empty((256,),device=rank,dtype=torch.int32),torch.empty((256,),device=rank,dtype=torch.int32),torch.empty((1,),device=rank,dtype=torch.int32)];partial=torch.empty((256,3584),device=rank,dtype=torch.float32)
   wp=[ptr(weights[k])for k in ['w13','s13','w2','s2']];baseargs=[rank,4,ptr(y),ptr(activation),ptr(x),ptr(ids),ptr(routeweights),*wp,*[ptr(v)for v in workspace],ptr(partial),896,16,stream]
   coldargs=[ctx,16,ptr(y),ptr(activation),ptr(x),ptr(ids),ptr(routeweights),*wp,ptr(workspace[0]),ptr(workspace[1]),ptr(active),ptr(count),ptr(bank),ptr(packX),ptr(gemm)]
   def invoke(on):check(cold(*coldargs)if on else original(*baseargs))
   invoke(False);torch.cuda.synchronize();baseA,baseY=bits(activation),bits(y)
   invoke(True);torch.cuda.synchronize();candA,candY=bits(activation),bits(y);row['candidate_vs_current']={}
   for key,a,b in [('activation',candA,baseA),('output',candY,baseY)]:
    aa,bb=fp(a),fp(b);assert np.isfinite(aa).all()and np.isfinite(bb).all();row['candidate_vs_current'][key]={'changed':int(np.count_nonzero(a!=b)),'maxabs':float(np.abs(aa-bb).max()),'nrmse':float(np.sqrt(np.mean((aa.astype(np.float64)-bb)**2))/max(1e-30,np.sqrt(np.mean(bb.astype(np.float64)**2))))}
   # Nonzero numericaldifference isdiagnostic, notafinalModeltokenverdict.
   row['samples']=[];startEvent,endEvent=[torch.cuda.Event(enable_timing=True)for _ in range(2)]
   for _ in range(3):invoke(False);invoke(True)
   torch.cuda.synchronize();row['free_memory_after_warmup']=torch.cuda.mem_get_info()[0]
   for pair in range(8):
    for on in ([False,True]if pair%2==0 else[True,False]):
     torch.cuda.synchronize();tick=time.perf_counter();startEvent.record();invoke(on);endEvent.record();endEvent.synchronize()
     row['samples'].append({'pair':pair,'candidate':on,'complete_provider_GPU_event_ms':startEvent.elapsed_time(endEvent),'complete_eager_host_ms':(time.perf_counter()-tick)*1000})
   invoke(False);torch.cuda.synchronize();row['baseline_ownrepeat']={'activation':bool(np.array_equal(bits(activation),baseA)),'output':bool(np.array_equal(bits(y),baseY))}
   invoke(True);torch.cuda.synchronize();row['candidate_ownrepeat']={'activation':bool(np.array_equal(bits(activation),candA)),'output':bool(np.array_equal(bits(y),candY))}
   assert all(row['baseline_ownrepeat'].values())and all(row['candidate_ownrepeat'].values())
   row['scope']='Everycandidateinvocation reruns actualGPUroutepack/activeplan+4BcountD2Hsync+all32slotcolddequant/pack/GEMMF32C/restore+originalW2. Includespadding/lastchunk/miss, nofreerouteplan/noBF16residentcache. Measuredwholecompoundkernelprovider, notModel.'
   d['fixtures'].append(row);save();del x,ids,routeweights,activation,y,workspace,partial,startEvent,endEvent
  check(lib.kimi_bf16_cold_destroy_r8666(ctx));ctx=None
  del weights,host,vendor,bank,packX,gemm,active,count;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();d['normal_CUDA_cleanup']=True;save()
 except BaseException as e:d['error']=repr(e);save();raise
 finally:
  if ctx:check(lib.kimi_bf16_cold_destroy_r8666(ctx))
if __name__=='__main__':
 import sys
 manifest,output=sys.argv[1:];mp.spawn(worker,args=(manifest,output),nprocs=8,join=True);rows=[json.loads(pathlib.Path(output+'-rank'+str(i)+'.json').read_text())for i in range(8)];assert all(v['normal_CUDA_cleanup']and'error'not in v for v in rows);pathlib.Path(output).write_text(json.dumps({'all8normal':True,'scope':'ColdinclusiveBF16W13twofixtureactualcheckpointcomponentonly; notModeltokens/TPS','workers':rows},indent=2))
