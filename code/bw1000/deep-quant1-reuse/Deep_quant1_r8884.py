"""Eight concurrent TP0 replicas of retained real routed component. NOT TP8 or Model."""
import pathlib,os,json,hashlib,ctypes,time,traceback,collections,threading,concurrent.futures
import torch,numpy as np
from safetensors import safe_open
from deep_affinity_env_contract_r7166 import validate
assert validate(os.environ)['requested_width']==4
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=r/'results'/('deep-quant1-r8884-'+tag+'.json');assert not out.exists()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();hash_tensor=lambda t:hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
m=json.loads((r/'incoming/Deep-quant1-manifest-r8884.json').read_text());bp=r/m['cpu_receipt_path'];build=json.loads(bp.read_text());assert build['completed']and build['CPU_only'];assert sha(bp)==m['existing_remote_SHA256'][m['cpu_receipt_path']]
library=pathlib.Path(build['library']);assert library==r/'candidates/deep-quant1-r8883/libdeep_quant1_r8883.so';assert sha(library)==build['library_SHA256']
for u in build['compiled_units']:
 for p,h in u.get('deps',{}).items():assert sha(p)==h
for p,v in build['actual_DTK_bindings'].items():assert sha(p)==v['SHA256']
torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
lib=ctypes.CDLL(str(library));P=ctypes.c_void_p;I=ctypes.c_int
lib.deep_prefill_pack_create_r8750.argtypes=[I,ctypes.POINTER(P)];lib.deep_prefill_pack_create_r8750.restype=P
lib.deep_prefill_pack_forward_r8750.argtypes=[P,I,ctypes.c_size_t,ctypes.POINTER(P)];lib.deep_prefill_pack_forward_r8750.restype=I
lib.deep_prefill_pack_destroy_r8750.argtypes=[P,ctypes.POINTER(I)];lib.deep_prefill_pack_destroy_r8750.restype=I
lib.deep_prefill_pack_counts_r8767.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.deep_prefill_pack_counts_r8767.restype=I
lib.deep_prefill_pack_report_r8750.restype=ctypes.c_char_p;lib.deep_prefill_pack_error_r8750.restype=ctypes.c_char_p
lib.deep_quant1_quality_r8880.argtypes=[P,ctypes.c_size_t,ctypes.c_size_t,I,ctypes.POINTER(P)];lib.deep_quant1_quality_r8880.restype=I
class DI(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',P),('sname',ctypes.c_char_p),('saddr',P)]
dl=ctypes.CDLL(None);dl.dladdr.argtypes=[P,ctypes.POINTER(DI)];dl.dladdr.restype=I
providers={}
for symbol in ['deep_prefill_pack_forward_r8750','deep_quant1_quality_r8880','deep_prefill_pack_counts_r8767']:
 info=DI();assert dl.dladdr(ctypes.cast(getattr(lib,symbol),P),ctypes.byref(info));p=pathlib.Path(info.fname.decode()).resolve();providers[symbol]={'path':str(p),'SHA256':sha(p)};assert providers[symbol]['SHA256']==build['library_SHA256']
report={'bank_retire_defaultOFF':True,'memory_original_afterload_floor_bytes':6*1024**3,'memory_original_runtime_reserve_bytes':512*1024**2,'warm_ABBA_BAAB_restore':'A/Q1/F warm then A,Q1,F,F,Q1,A,A,F,Q1,Q1,F,A,restoreA','completed':False,'step':tag,'scope':'Eight independent TP0 real591 T2048/U251; acceptedA PACKON/originalscatter+bankfences/activationOFF; Q1twoIndexSelect and Q1fusedbytecopy only quant1 reuse; original stream/quant2/GemmEx/math/drain, notTP8Model/decode','library_SHA256':sha(library),'CPU_receipt_SHA256':sha(bp),'providers':providers,'owners':[],'no_Model_TPS':True,'performance_adopted':False}
def save():
 q=out.with_suffix('.tmp');q.write_text(json.dumps(report,indent=2));q.replace(out)
def check(code):
 if code:raise RuntimeError(lib.deep_prefill_pack_error_r8750().decode())
started=time.perf_counter()
meta=r/'results/deep-layer0-r7005-910374.591-owner0-B32.json';assert sha(meta)=='977be1e7bdaef015744cc6d3921a73d5408d059ec99b16ebd0f7f2f64e42b37e';event=json.loads(meta.read_text())['history'][0];assert event['rows']==[0,1,2,3];stages={v['name']:v for v in event['stages']};selected=list(range(2048));inputs=[];proof=[]
for name,dtype in [('ffn_norm',np.uint16),('moe_indices',np.int64),('moe_weights',np.float32)]:
 s=stages[name];p=r/'results/deep-layer0-r7005-910374.591'/s['raw_file'];assert sha(p)==s['whole_SHA256'];a=np.fromfile(p,dtype=dtype).reshape(s['shape'])[selected].copy();t=torch.from_numpy(a).view(torch.bfloat16)if name=='ffn_norm'else torch.from_numpy(a);inputs.append(t);proof.append({'name':name,'source_SHA256':sha(p),'selected_SHA256':hash_tensor(t),'shape':list(t.shape)})
report['input_proof']=proof;report['selected_prefill_tensor_rows']=selected;report['input_prepare_seconds']=time.perf_counter()-started
model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];gold={v['key']:v['CPU_tensor_SHA256']for v in json.loads((r/'results/deep-actualprefix-r6994-910374.587-CPUweights.json').read_text())}
banks=[torch.empty((256,512,4096),dtype=torch.int8),torch.empty((256,512,1),dtype=torch.float32),torch.empty((256,4096,256),dtype=torch.int8),torch.empty((256,4096,1),dtype=torch.float32)]
groups=collections.defaultdict(list)
for e in range(256):
 for w in ['w1','w3','w2']:
  for suffix in ['weight','scale']:
   key=f'layers.0.ffn.experts.{e}.{w}.{suffix}';groups[mapping[key]].append((key,e,w,suffix))
weights_begin=time.perf_counter();weightproof=[];full_source_bytes=0
for filename,keys in sorted(groups.items()):
 with safe_open(model/filename,framework='pt',device='cpu')as f:
  for key,e,w,suffix in keys:
   t=f.get_tensor(key).contiguous();h=hash_tensor(t);assert h==gold[key],key;full_source_bytes+=t.numel()*t.element_size()
   if w in['w1','w3']:
    offset=0 if w=='w1' else 256;bank=banks[0 if suffix=='weight' else 1];bank[e,offset:offset+256].copy_(t[:256])
   elif suffix=='weight':banks[2][e].copy_(t[:,:256])
   else:banks[3][e].copy_(t)
   weightproof.append({'key':key,'SHA256':h})
report['CPU_checkpoint_read_hash_slice_seconds']=time.perf_counter()-weights_begin;report['full_source_tensor_bytes_read_hashed']=full_source_bytes;report['weight_keys']=len(weightproof);report['source_weight_manifest_SHA256']=hashlib.sha256(json.dumps(weightproof,sort_keys=True).encode()).hexdigest();report['TP0_bank_SHA256']=[hash_tensor(t)for t in banks];report['CPU_bank_bytes']=sum(t.numel()*t.element_size()for t in banks)
arms=[0,1,2]+[0,1,2,2,1,0,0,2,1,1,2,0]+[0];report.update(arms=arms,warm_trials=list(range(3)),formal_trials=list(range(3,15)),restore_trial=15);unique_experts=int(torch.unique(inputs[1]).numel());report.update(tokens=2048,routes=12288,unique_experts=unique_experts,metadata_H2D_bytes=196608,max_expert_rows=int(torch.bincount(inputs[1].reshape(-1),minlength=256).max()),extra_source_bank_bytes=8396800);save()
cpu_set=sorted(os.sched_getaffinity(0));assert len(cpu_set)>=32;barrier=threading.Barrier(8)
def counts(dev):
 a=(ctypes.c_uint64*12)();check(lib.deep_prefill_pack_counts_r8767(dev,a,12));return list(a)
def owner(dev):
 ctx=None;records=[];d={'dev':dev,'records':records,'normal_owner_closed':False};op=r/'results'/f'Deep-quant1-r8884-{tag}-owner{dev}.json'
 def writeowner():op.write_text(json.dumps(d))
 try:
  torch.cuda.set_device(dev);props=torch.cuda.get_device_properties(dev);assert 'gfx936'in getattr(props,'gcnArchName','')and props.multi_processor_count==80 and props.total_memory>=63*1024**3
  assert torch.cuda.mem_get_info(dev)[0]>=8*1024**3
  h2d=time.perf_counter();gpu_inputs=[t.to(dev)for t in inputs];gpu_banks=[t.to(dev)for t in banks];output=torch.empty((2048,4096),device=dev,dtype=torch.float32);torch.cuda.synchronize(dev);d['GPU_bank_input_H2D_seconds']=time.perf_counter()-h2d
  tensors=[*gpu_inputs,output];ptrs=(P*4)(*[t.data_ptr()for t in tensors]);bankptrs=(P*4)(*[t.data_ptr()for t in gpu_banks]);ctx=lib.deep_prefill_pack_create_r8750(dev,bankptrs);assert ctx,lib.deep_prefill_pack_error_r8750().decode();d['affinity_before_first_submit']=sorted(os.sched_getaffinity(0));assert set(d['affinity_before_first_submit'])==set(cpu_set);d['hardware']={'arch':props.gcnArchName,'CU':props.multi_processor_count,'total_memory':props.total_memory};d['free_after_load']=torch.cuda.mem_get_info(dev)[0];assert d['free_after_load']>=6*1024**3
  # Outside timing: actual whole-source quant vs original per-expert quant,
  # and CPU byte-copy oracle from identity rows, including repeated row indices.
  quality=[]
  for kind in ['actual','finite_edges','nan_inf']:
   if kind=='actual':host=inputs[0];source=gpu_inputs[0]
   else:
    vals=torch.tensor([-127,-64,-1.5,-.5,-0.0,0.0,.5,1.5,63.5,127,2**-126,-2**-126],dtype=torch.bfloat16)
    host=vals.repeat((8*4096+len(vals)-1)//len(vals))[:8*4096].reshape(8,4096).contiguous()
    host[0].zero_()
    if kind=='nan_inf':host[2,0]=float('nan');host[3,1]=float('inf');host[4,2]=-float('inf')
    source=host.to(dev)
   T=source.shape[0];basepacked=None;basescale=None
   for selected in [torch.arange(T,dtype=torch.int64),torch.tensor([0,T-1,0,1,T//2,T//2],dtype=torch.int64)]:
    ids=selected.to(dev);N=len(selected)
    ref=torch.empty((N,4096),dtype=torch.int8,device=dev);refscale=torch.empty((N,1),dtype=torch.float32,device=dev);got=torch.empty_like(ref);gotscale=torch.empty_like(refscale)
    for fused in [0,1]:
     qp=(P*6)(*[t.data_ptr() for t in [source,ids,ref,refscale,got,gotscale]])
     check(lib.deep_quant1_quality_r8880(ctx,T,N,fused,qp));rp=ref.cpu();rs=refscale.cpu();gp=got.cpu();gs=gotscale.cpu()
     assert torch.equal(rp,gp) and torch.equal(rs.view(torch.int32),gs.view(torch.int32)),('quant1_gather_byte_scale_difference',kind,fused)
     if N==T:
      basepacked=rp.clone();basescale=rs.clone()
     else:assert torch.equal(gp,basepacked[selected]) and torch.equal(gs.view(torch.int32),basescale[selected].view(torch.int32)),('CPU_rowcopy_oracle',kind,fused)
     quality.append({'kind':kind,'rows':N,'fused':fused,'packed_and_scale_bytes_exact':True,'CPU_duplicate_rowcopy_oracle':N!=T})
  d['quant1_boundary_actual_quality']=quality
  d['free_after_quality_prepare']=torch.cuda.mem_get_info(dev)[0];assert d['free_after_quality_prepare']>=6*1024**3
  gold=None
  for trial,mode in enumerate(arms):
   barrier.wait(timeout=180);count_before=counts(dev);free_before=torch.cuda.mem_get_info(dev)[0];assert free_before>=512*1024**2;begin=time.perf_counter();cpu=time.thread_time();check(lib.deep_prefill_pack_forward_r8750(ctx,mode,2048,ptrs));end=time.perf_counter();cpuseconds=time.thread_time()-cpu;native=json.loads(lib.deep_prefill_pack_report_r8750().decode());d['affinity']=sorted(os.sched_getaffinity(0));assert set(d['affinity'])==set(cpu_set[4*dev:4*dev+4]);assert native['mode']==mode and native['tokens']==2048 and native['routes']==12288 and native['unique_experts']==unique_experts and native['metadata_H2D_calls']==2 and native['metadata_H2D_syncs']==1 and native['metadata_packed'] and not native['scatter_retire'] and not native['bank_retire'] and native['bank_success_fence_calls_source_contract']==unique_experts and native['scatter_post_fences']==unique_experts and native['scatter_post_fences_skipped']==0 and native['metadata_H2D_bytes']==196608
   assert native['quant1_calls']==int(mode>0) and native['quant1_rows']==(2048 if mode else 0) and native['gather_scale_calls']==(unique_experts if mode else 0) and native['gather_route_rows']==(12288 if mode else 0)
   count_delta=[b-a for a,b in zip(count_before,counts(dev))];expected=[0,1,0,unique_experts,0,2,0,1,0,196608,0,0];assert count_delta==expected;
   a=output.cpu().numpy();assert np.isfinite(a).all()
   if gold is None:gold=a.copy()
   delta=a.astype(np.float64)-gold.astype(np.float64);maxabs=float(np.max(np.abs(delta)));nrmse=float(np.sqrt(np.mean(delta*delta))/max(float(np.sqrt(np.mean(gold.astype(np.float64)**2))),1e-30));numeric=bool(np.allclose(a,gold,rtol=1e-5,atol=1e-5));abspercentiles=np.percentile(np.abs(delta),[0,50,95,99,100]).tolist();argmaxdiff=int(np.count_nonzero(np.argmax(a,axis=1)!=np.argmax(gold,axis=1)))
   free_after=torch.cuda.mem_get_info(dev)[0];assert free_after>=512*1024**2;d['observed_memory_floor_bytes']=min(d.get('observed_memory_floor_bytes',d['free_after_load']),free_before,free_after);d['memory_scope']='afterload/beforecall/aftercall snapshots; notduringcallpeak';z=dict(native,quant1_mode=mode,free_before_call_bytes=free_before,free_after_call_bytes=free_after,phase_counter_delta=count_delta,trial=trial,warm=trial<3,formal=3<=trial<15,complete_CAPI_wall_seconds=end-begin,owner_thread_CPU_seconds=cpuseconds,CPU_CAPI_begin=begin,CPU_CAPI_end=end,finite=True,numeric_guard=numeric,maxabs=maxabs,NRMSE=nrmse,absolute_error_percentiles_0_50_95_99_100=abspercentiles,argmax_rows_different_diagnostic=argmaxdiff,byteexact_diagnostic=bool(np.array_equal(a,gold)),output_SHA256=hashlib.sha256(a.tobytes()).hexdigest());records.append(z);writeowner()
   assert numeric,('full_routed_output_difference',dev,trial,maxabs,nrmse);assert torch.cuda.mem_get_info(dev)[0]>=512*1024**2
  return d
 except BaseException as e:d['error']=repr(e);d['traceback']=traceback.format_exc();barrier.abort();raise
 finally:
  if ctx:
   expired=I();d['destroy_code']=lib.deep_prefill_pack_destroy_r8750(ctx,ctypes.byref(expired));d['Runtime_expired']=expired.value;d['normal_owner_closed']=d['destroy_code']==0 and expired.value==1
  writeowner()
  if ctx:assert d['normal_owner_closed']
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:report['owners']=list(pool.map(owner,range(8)))
 masks=[set(o['affinity'])for o in report['owners']];assert len(set.union(*masks))==32;assert all(o['normal_owner_closed']for o in report['owners']);report['completed']=True
except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc());raise
finally:report['driver_wall_includes_CPUweights_H2D_readback_seconds']=time.perf_counter()-started;save()
print(json.dumps({'completed':report['completed'],'report':str(out)}))
