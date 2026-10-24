"""Actual layer3 NativeDecoder component; embedding-derived hidden, not Model replay/TPS."""
import ctypes,hashlib,json,os,pathlib,threading,concurrent.futures,time,statistics
import numpy as np
import torch
from safetensors import safe_open
from deep_affinity_env_contract_r7166 import validate
assert validate(os.environ)['requested_width']==4
torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=r/'results'/f'Deep-position-layer-r8731-{tag}.json';assert not path.exists();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();hashraw=lambda b:hashlib.sha256(b).hexdigest();ptr=lambda t:ctypes.c_void_p(t.data_ptr())
bp=r/'results/deep-layer3-position-fixture-r8732-build.json';build=json.loads(bp.read_text());assert build['completed']and build['CPU_only'];library=pathlib.Path(build['library']);assert library==r/'candidates/deep-layer3-position-fixture-r8732/libdeep_layer3_position_fixture_r8732.so';assert sha(library)==build['library_SHA256']
manifest=json.loads((r/'incoming/Deep-position-layer-manifest-r8731.json').read_text());assert sha(bp)==manifest['existing_remote_SHA256']['results/deep-layer3-position-fixture-r8732-build.json']
for n,h in build['deps'].items():assert sha(n)==h
for n,h in build['link']['objects'].items():assert sha(n)==h
for n,v in build['actual_DTK_bindings'].items():assert sha(n)==v['SHA256']
lib=ctypes.CDLL(str(library))
class DLInfo(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('fbase',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
dladdr=ctypes.CDLL(None).dladdr;dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DLInfo)];dladdr.restype=ctypes.c_int
for symbol in ['deep_partial_forward_r7003','deep_position_proof_counts_r8726']:
 info=DLInfo();assert dladdr(ctypes.cast(getattr(lib,symbol),ctypes.c_void_p),ctypes.byref(info))==1;assert pathlib.Path(info.fname.decode()).resolve()==library.resolve()
P=ctypes.c_void_p;C=ctypes.c_char_p;Z=ctypes.c_size_t;I=ctypes.c_int;U=ctypes.c_uint64
spec={'error':([],C),'id_size':([],Z),'id':([P],I),'create':([I,C,P],P),'keys':([P],C),'load':([P,C,P,Z,P,I],I),'finish':([P],I),'reset':([P],I),'forward':([P,Z,Z]+[P]*6,I),'stages':([P],C),'read':([P,Z,P,Z],I),'destroy':([P],I)}
for n,(a,z)in spec.items():f=getattr(lib,'deep_partial_'+n+'_r7003');f.argtypes=a;f.restype=z

def check(q):
 if q:raise RuntimeError(lib.deep_partial_error_r7003().decode())
for n,v in [('modern_deep_precision_policy_r3443',5),('modern_deep_allselected_set_r813',1),('modern_deep_layout_policy_r7052',0),('deep_hc_canonical_policy_r7130',0),('deep_moe_tail_set_r7211',0),('modern_deep_c4_tile_policy_r5920',16)]:
 f=getattr(lib,n);f.argtypes=[I];f.restype=I;check(f(v))
lib.deep_position_proof_counts_r8726.argtypes=[I,ctypes.POINTER(U),Z];lib.deep_position_proof_counts_r8726.restype=I
os.environ['INFINILM_DEEP_OUTPUT_TILE_R4808']='16';os.environ['INFINILM_DEEP_OUTPUT_DIRECT_R7247']='0';os.environ['INFINILM_DEEP_POSITION_PROOF_R8726']='0'
model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');cfg=json.loads((model/'config.json').read_text());assert cfg['compress_ratios'][3]==128 and cfg['num_hidden_layers']==43;mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];preq=json.loads((r/'results/deep-actualprefix-r6993-preflight.json').read_text());assert sha(model/'config.json')==preq['config_SHA256']=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09';assert sha(model/'model.safetensors.index.json')==preq['checkpoint_index_SHA256']=='7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a'
paramp=r/'results/deep-actualprefix-r6993-parameters.json';assert sha(paramp)==preq['source_index_SHA256'];params=[v for v in json.loads(paramp.read_text())['parameters']if v['key'].startswith('layers.3.')];expected={v['key']for v in params};assert len(expected)==1568
natural=json.loads((r/'results'/f'deep-serving-cases-r2130-{tag}.json').read_text());assert len(natural['cases'])==2 and all(len(v['input_ids'])==512 for v in natural['cases']);prompts=[natural['cases'][i%2]['input_ids']for i in range(32)]
# Extend legal teacher inputs using these known checkpoint vocabulary IDs, without claiming Model-generated layer3 state.
seqs=[(p+p)[:640]for p in prompts];weights={};grouped={};weightproof=[];start=time.perf_counter()
for row in params:grouped.setdefault(row['file'],[]).append(row)
for fn,rows in sorted(grouped.items()):
 with safe_open(model/fn,framework='pt',device='cpu')as f:
  for row in rows:
   k=row['key'];t=f.get_tensor(k).contiguous();assert list(t.shape)==row['shape'];weights[k]=t;weightproof.append({'key':k,'shape':list(t.shape),'dtype':str(t.dtype),'CPU_tensor_SHA256':hashraw(memoryview(t.reshape(-1).view(torch.uint8).numpy()))})
dtype_codes={torch.int8:0,torch.float32:1,torch.bfloat16:2,torch.int64:3};assert set(weights)==expected and all(w.dtype in dtype_codes for w in weights.values());unique=sorted({t for p in seqs for t in p});index={t:i for i,t in enumerate(unique)}
with safe_open(model/mapping['embed.weight'],framework='pt',device='cpu')as f:emb=torch.stack([f.get_slice('embed.weight')[t:t+1][0]for t in unique]).contiguous()
arms=[0,1,0,1,1,0,0,1,1,0];barrier=threading.Barrier(8);cpu_set=sorted(os.sched_getaffinity(0));assert len(cpu_set)>=32;uid=ctypes.create_string_buffer(lib.deep_partial_id_size_r7003());assert lib.deep_partial_id_size_r7003()==136;check(lib.deep_partial_id_r7003(uid));rawroot=r/'results'/f'Deep-position-layer-r8731-{tag}-raw';rawroot.mkdir()
report={'completed':False,'step':tag,'scope':'Actual checkpoint NativeDecoder layer3 C128 TP8/B32, sameRuntime one load. Embedding-derived residual and legal repeated-natural teacher IDs are synthetic layer inputs, not699 Model hidden/logits/tokens. Only forward-local positions D2H reuse changes. No Model TPS or final-token claim.','CPU_weight_seconds':time.perf_counter()-start,'weight_manifest_SHA256':hashraw(json.dumps(weightproof,sort_keys=True).encode()),'weights_count':len(weights),'library_SHA256':sha(library),'build_SHA256':sha(bp),'prefixes':[575,639],'arms':arms,'warm_trials':[0,1],'formal_trials':list(range(2,10)),'performance_adopted':False,'owners':[]}

def save():
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
save()
def counts(dev):
 out=(U*3)();check(lib.deep_position_proof_counts_r8726(dev,out,3));return list(out)
def owner(dev):
 ctx=None;records=[];baseline={};mask=None;ownerpath=r/'results'/f'Deep-position-layer-r8731-{tag}-owner{dev}.json'
 try:
  inherited=sorted(os.sched_getaffinity(0));torch.cuda.set_device(dev);ctx=lib.deep_partial_create_r7003(dev,json.dumps(cfg).encode(),uid);assert ctx,lib.deep_partial_error_r7003();assert set(json.loads(lib.deep_partial_keys_r7003(ctx)))==expected
  for name in json.loads(lib.deep_partial_keys_r7003(ctx)):
   w=weights[name];shape=(Z*w.ndim)(*w.shape);check(lib.deep_partial_load_r7003(ctx,name.encode(),ptr(w),w.ndim,shape,dtype_codes[w.dtype]))
  check(lib.deep_partial_finish_r7003(ctx));assert torch.cuda.mem_get_info(dev)[0]>=6*1024**3;mask=sorted(os.sched_getaffinity(0));assert len(mask)==4 and set(mask)<=set(inherited);barrier.wait(timeout=600)
  def forward(rows,past,length,timed):
   chunks=[seqs[i][past:past+length]for i in rows];ids=sum(chunks,[]);pos=sum([list(range(past,past+length))for i in rows],[]);req=np.ascontiguousarray([[i,past,length]for i in rows],dtype=np.uint64);x=emb[[index[t]for t in ids]].unsqueeze(1).expand(-1,4,-1).contiguous().to(dev);ti=torch.tensor(ids,dtype=torch.int64,device=dev);tp=torch.tensor(pos,dtype=torch.int64,device=dev);before=counts(dev);barrier.wait(timeout=180);tick=time.perf_counter();cpu=time.thread_time();check(lib.deep_partial_forward_r7003(ctx,len(ids),len(rows),ctypes.c_void_p(req.ctypes.data),ptr(x),ptr(ti),ptr(tp),None,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)));wall=time.perf_counter()-tick;cpu=time.thread_time()-cpu;after=counts(dev);delta=[b-a for a,b in zip(before,after)];on=os.environ['INFINILM_DEEP_POSITION_PROOF_R8726']=='1';assert delta==([1,len(ids)*8,0]if on else[0,0,1]),('actual_proof_hit_or_fallback',delta,on,len(ids));assert torch.cuda.mem_get_info(dev)[0]>=512*1024**2;return {'tokens':len(ids),'counter_delta':delta,'CAPI_wall_seconds':wall,'owner_thread_CPU_seconds':cpu,'CPU_CAPI_begin':tick,'CPU_CAPI_end':tick+wall}
  for past in [575,639]:
   for trial,flag in enumerate(arms):
    barrier.wait(timeout=180)
    if dev==0:os.environ['INFINILM_DEEP_POSITION_PROOF_R8726']=str(flag)
    barrier.wait(timeout=180);check(lib.deep_partial_reset_r7003(ctx));seed_start=time.perf_counter();seed=[]
    for first in range(0,32,4):seed.append(forward(list(range(first,first+4)),0,past,False))
    seed_wall=time.perf_counter()-seed_start;result=forward(list(range(32)),past,1,True);metas=json.loads(lib.deep_partial_stages_r7003(ctx));assert len(metas)==17;outputs=[]
    for si,meta in enumerate(metas):
     assert meta['dtype']in['BF16','F32','I32','I64'];buf=(ctypes.c_ubyte*meta['bytes'])();check(lib.deep_partial_read_r7003(ctx,si,buf,meta['bytes']));raw=bytes(buf);key=(past,si);arr=np.frombuffer(raw,{'BF16':np.uint16,'F32':np.float32,'I32':np.int32,'I64':np.int64}[meta['dtype']]);v=(arr.astype(np.uint32)<<16).view(np.float32)if meta['dtype']=='BF16'else arr;assert np.isfinite(v).all()
     if trial==0:baseline[key]=raw;(rawroot/f'owner{dev}-P{past}-{si}-O.bin').write_bytes(raw)
     old=baseline[key];oldarr=np.frombuffer(old,arr.dtype);ov=(oldarr.astype(np.uint32)<<16).view(np.float32)if meta['dtype']=='BF16'else oldarr;exact=raw==old;maxabs=float(np.max(np.abs(v.astype(np.float64)-ov.astype(np.float64))))if len(v)else 0.;numeric=bool(np.array_equal(v,ov))if meta['dtype']in['I32','I64']else bool(np.allclose(v,ov,rtol=1e-3,atol=1e-3));outputs.append({'name':meta['name'],'shape':meta['shape'],'dtype':meta['dtype'],'bytes':len(raw),'SHA256':hashraw(raw),'OFF0_exact_diagnostic':exact,'numeric_guard':numeric,'maxabs':maxabs})
     if not numeric:(rawroot/f'owner{dev}-P{past}-{si}-trial{trial}-failed.bin').write_bytes(raw)
     assert numeric,('actual_whole_layer_output_difference',dev,past,trial,meta['name'],maxabs)
    records.append({'past':past,'trial':trial,'ON':bool(flag),'warm':trial<2,'seed_wall_seconds':seed_wall,'seed':seed,'decode':result,'all17_outputs':outputs});ownerpath.write_text(json.dumps({'dev':dev,'records':records,'normal_owner_closed':False}));barrier.wait(timeout=180)
  check(lib.deep_partial_reset_r7003(ctx));return {'dev':dev,'affinity':mask,'records':records,'normal_owner_closed':True}
 except BaseException:
  barrier.abort();raise
 finally:
  if ctx:check(lib.deep_partial_destroy_r7003(ctx))
  torch.cuda.synchronize(dev)
  ownerpath.write_text(json.dumps({'dev':dev,'records':records,'normal_owner_closed':True}))
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:report['owners']=list(pool.map(owner,range(8)))
 masks=[set(o['affinity'])for o in report['owners']];assert all(len(v)==4 for v in masks)and len(set.union(*masks))==32;os.environ['INFINILM_DEEP_POSITION_PROOF_R8726']='0';report['owner_masks_disjoint_verified']=True;report['completed']=True;report['no_Model_TPS']=True;save();print(json.dumps({'completed':True,'report':str(path)}))
except BaseException as e:report['error']=repr(e);save();raise

