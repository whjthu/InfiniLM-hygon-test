"""Actual rank0 checkpoint provider; original versus invalid-second-row arithmetic guard."""
import pathlib,json,hashlib,ctypes,time,contextlib,gc,sys,os
import numpy as np
import torch
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/shared/Kimi-K3');L=1;BANK=0
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(manifest,output):
 rank=0;torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
 m=json.loads(pathlib.Path(manifest).read_text());assert sha(m['library'])==m['library_SHA256'];lib=ctypes.CDLL(m['library'])
 original=lib.modern_kimi_grouped_r4620;original.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*15+[ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p]
 candidate=lib.modern_kimi_valid_events_r8669i;candidate.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*15+[ctypes.c_size_t]*2+[ctypes.c_void_p]*3
 out=pathlib.Path(output);assert not out.exists();d={'rank':0,'visible_devices':1,'normal_CUDA_cleanup':False,'scope':'ActualTP8rank0L1shard, ordinarycomponentpairedtimings; noModelclaim','candidate':'invalidsecondFMAuniformouterbranch','Model_performance_claim':False}
 def save():
  q=out.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(out)
 def ptr(t):return ctypes.c_void_p(t.data_ptr())
 def bits(t):return t.detach().cpu().view(torch.uint16).numpy().copy()
 def fp(v):return(v.astype(np.uint32)<<16).view(np.float32)
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


  props=torch.cuda.get_device_properties(0);d['arch']=getattr(props,'gcnArchName','');assert 'gfx936'in d['arch'];d['device_name']=props.name
  refs=m['captured_fixture_refs']['0'];actual={}
  for point,z in refs.items():
   f=pathlib.Path(z['file']);assert sha(f)==z['SHA256'];actual[point]=np.fromfile(f,{'I32':np.int32,'F32':np.float32,'BF16':np.uint16}[z['dtype']]).reshape(z['shape'])
  rng=np.random.default_rng(4552);seedX=torch.from_numpy(rng.standard_normal((16,3584),dtype=np.float32)*.1).to(torch.bfloat16);seedIDs=np.stack([rng.choice(896,16,replace=False).astype(np.int32)for _ in range(16)]);seedRW=rng.uniform(.01,1,(16,16)).astype(np.float32);seedRW/=seedRW.sum(-1,keepdims=True)
  fixtures=[('captured21',torch.from_numpy(np.tile(actual['routed_input'],(2,1)).copy()).view(torch.bfloat16),np.tile(actual['routing_ids'],(2,1)).copy(),np.tile(actual['routing_weights'],(2,1)).copy()),('seed229',seedX,seedIDs,seedRW)]
  d['fixtures']=[];stream=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
  for index,(label,x_cpu,ids_cpu,rw)in enumerate(fixtures):
   assert all(len(np.unique(v))==16 for v in ids_cpu);active=len(np.unique(ids_cpu));assert active==(21 if index==0 else 229)
   row={'label':label,'active_experts':active,'T':16,'fixture_SHA256':{'input':hashlib.sha256(x_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),'ids':hashlib.sha256(ids_cpu.tobytes()).hexdigest(),'routes':hashlib.sha256(rw.tobytes()).hexdigest()}}
   counts=np.bincount(ids_cpu.reshape(-1),minlength=896);jobs=(counts+1)//2;partial_jobs=int(np.count_nonzero(counts%2));row['TT2_jobs']={'full':int(jobs.sum())-partial_jobs,'partial':partial_jobs,'total':int(jobs.sum()),'real_routes':int(counts.sum()),'allocated_slots':int(2*jobs.sum()),'slot_fill':float(counts.sum()/(2*jobs.sum()))}
   x=x_cpu.cuda();ids=torch.from_numpy(ids_cpu).cuda();routeweights=torch.from_numpy(rw).cuda();activation=torch.empty((16,16,384),device=0,dtype=torch.bfloat16);y=torch.empty_like(x)
   workspace=[torch.empty((896,),device=0,dtype=torch.int32),torch.empty((896,256),device=0,dtype=torch.int32),torch.empty((256,),device=0,dtype=torch.int32),torch.empty((256,),device=0,dtype=torch.int32),torch.empty((1,),device=0,dtype=torch.int32)];partial=torch.empty((256,3584),device=0,dtype=torch.float32)
   args=[0,4,ptr(y),ptr(activation),ptr(x),ptr(ids),ptr(routeweights),*[ptr(weights[k])for k in['w13','s13','w2','s2']],*[ptr(v)for v in workspace],ptr(partial),896,16,stream]
   events=[torch.cuda.Event(enable_timing=True)for _ in range(4)]
   for ev in events:ev.record()
   torch.cuda.synchronize();begin,end,wb,we=events
   def invoke(mode,with_events=False):
    rc=candidate(0,4,mode,*args[2:],ctypes.c_void_p(wb.cuda_event)if with_events else None,ctypes.c_void_p(we.cuda_event)if with_events else None);assert rc==0,rc
   invoke(0);torch.cuda.synchronize();refA,refY=bits(activation),bits(y);assert np.isfinite(fp(refA)).all()and np.isfinite(fp(refY)).all()
   invoke(1);torch.cuda.synchronize();row['mode1_faithful_mode0']={'activation':bool(np.array_equal(bits(activation),refA)),'output':bool(np.array_equal(bits(y),refY))};assert all(row['mode1_faithful_mode0'].values())
   invoke(2);torch.cuda.synchronize();candA,candY=bits(activation),bits(y);row['candidate_vs_original']={}
   for key,a,b in[('activation',candA,refA),('output',candY,refY)]:
    aa,bb=fp(a),fp(b);assert np.isfinite(aa).all()and np.isfinite(bb).all();row['candidate_vs_original'][key]={'changed':int(np.count_nonzero(a!=b)),'maxabs':float(np.abs(aa-bb).max()),'nrmse':float(np.linalg.norm(aa.astype(np.float64)-bb)/max(1e-30,np.linalg.norm(bb.astype(np.float64))))}
   # Candidate intermediate bits diagnostic; only independent control must faithfully reproduce original.
   for _ in range(3):invoke(1);invoke(2)
   torch.cuda.synchronize();row['original_fullprovider_reference_samples']=[]
   for _ in range(3):
    start=time.perf_counter();begin.record();invoke(0);end.record();end.synchronize();row['original_fullprovider_reference_samples'].append({'GPU_ms':begin.elapsed_time(end),'host_ms':(time.perf_counter()-start)*1000,'scope':'SameoriginalAPI referenceonly; notW13-onlytiming'})
   row['paired_samples']=[]
   for pair in range(8):
    for mode in([1,2]if pair%2==0 else[2,1]):
     torch.cuda.synchronize();start=time.perf_counter();begin.record();invoke(mode,True);end.record();end.synchronize();row['paired_samples'].append({'pair':pair,'mode':mode,'fullprovider_GPU_ms':begin.elapsed_time(end),'W13_GPU_ms':wb.elapsed_time(we),'fullprovider_host_ms':(time.perf_counter()-start)*1000})
   row['ownrepeat']={}
   for mode,aa,bb in[(0,refA,refY),(1,refA,refY),(2,candA,candY)]:
    invoke(mode);torch.cuda.synchronize();same=bool(np.array_equal(bits(activation),aa)and np.array_equal(bits(y),bb));row['ownrepeat'][str(mode)]=same;assert same
   d['fixtures'].append(row);save();del x,ids,routeweights,activation,y,workspace,partial,events,begin,end,wb,we
  del weights,host;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();d['normal_CUDA_cleanup']=True;d['completed']=True;save()
 except BaseException as e:d['error']=repr(e);save();raise
if __name__=='__main__':run(*sys.argv[1:])
