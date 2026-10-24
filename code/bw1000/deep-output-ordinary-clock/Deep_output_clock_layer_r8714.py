"""Existing r7003 stateful layer0 fixture. Whole-layer times only; no serving TPS."""
import ctypes,hashlib,json,os,pathlib,threading,concurrent.futures,time,statistics
import numpy as np
import torch
from safetensors import safe_open
from deep_affinity_env_contract_r7166 import validate
assert validate(os.environ)['requested_width']==4
torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=r/'results'/f'Deep-output-clock-layer-r8716-{tag}.json';assert not path.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashraw=lambda b:hashlib.sha256(b).hexdigest();ptr=lambda t:ctypes.c_void_p(t.data_ptr())
bp=r/'results/deep-output-clock-r8713-build.json';assert sha(bp)=='b67915b4d15d44aac1c97385c28dcc40e5c4055cbca205605dc54173f59521a9';build=json.loads(bp.read_text());assert build['completed'];library=pathlib.Path(build['libraries']['partial']['path']);assert sha(library)=='0e2e617bc0b194708e9ec8855d2a0bb5fd200848ac72e305d7c8cbc18935e546'
for n,h in build['compiled_unit']['deps'].items():assert sha(pathlib.Path(n))==h
for n,h in build['link']['partial']['objects'].items():assert sha(pathlib.Path(n))==h
for n,v in build['actual_DTK_bindings'].items():assert sha(pathlib.Path(n))==v['SHA256']
lib=ctypes.CDLL(str(library))
class DLInfo(ctypes.Structure):
 _fields_=[('fname',ctypes.c_char_p),('fbase',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
dladdr=ctypes.CDLL(None).dladdr;dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DLInfo)];dladdr.restype=ctypes.c_int
for symbol in ['deep_partial_forward_r7003','deep_output_clock_config_r8713']:
 info=DLInfo();assert dladdr(ctypes.cast(getattr(lib,symbol),ctypes.c_void_p),ctypes.byref(info))==1;assert sha(pathlib.Path(info.fname.decode()))==sha(library)
P=ctypes.c_void_p;C=ctypes.c_char_p;Z=ctypes.c_size_t;I=ctypes.c_int;U=ctypes.c_uint64
spec={'error':([],C),'id_size':([],Z),'id':([P],I),'create':([I,C,P],P),'keys':([P],C),'load':([P,C,P,Z,P,I],I),'finish':([P],I),'reset':([P],I),'forward':([P,Z,Z]+[P]*6,I),'stages':([P],C),'read':([P,Z,P,Z],I),'destroy':([P],I)}
for n,(a,z)in spec.items():f=getattr(lib,'deep_partial_'+n+'_r7003');f.argtypes=a;f.restype=z

def check(q):
 if q:raise RuntimeError(lib.deep_partial_error_r7003().decode())
for n in ['modern_deep_precision_policy_r3443','modern_deep_allselected_set_r813','modern_deep_layout_policy_r7052','deep_hc_canonical_policy_r7130','deep_moe_tail_set_r7211','modern_deep_c4_tile_policy_r5920']:
 getattr(lib,n).argtypes=[I];getattr(lib,n).restype=I
for n,value in [('modern_deep_precision_policy_r3443',5),('modern_deep_allselected_set_r813',1),('modern_deep_layout_policy_r7052',0),('deep_hc_canonical_policy_r7130',0),('deep_moe_tail_set_r7211',0),('modern_deep_c4_tile_policy_r5920',16)]:check(getattr(lib,n)(value))
lib.deep_output_clock_config_r8713.argtypes=[I,I];lib.deep_output_clock_config_r8713.restype=I
lib.deep_output_clock_read_r8713.argtypes=[ctypes.POINTER(ctypes.c_double),Z];lib.deep_output_clock_read_r8713.restype=I
lib.deep_output_tile_calls_r4808.argtypes=[I];lib.deep_output_tile_calls_r4808.restype=U
os.environ['INFINILM_DEEP_OUTPUT_TILE_R4808']='16';os.environ['INFINILM_DEEP_OUTPUT_DIRECT_R7247']='0'
for n in ['INFINILM_DEEP_LAYER_TRACE_R4026','DEEP_CSA_CAPTURE_R5800']:os.environ.pop(n,None)
provider={}
for needle in ['libinfiniops.so','libinfinilm_runtime.so']:
 maps=sorted({v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if needle in v});assert len(maps)==1
 expected=next(v['SHA256']for n,v in build['actual_DTK_bindings'].items()if n.endswith('/'+needle));assert sha(pathlib.Path(maps[0]))==expected;provider[needle]={'path':maps[0],'SHA256':expected}
model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');cfg=json.loads((model/'config.json').read_text());mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];preq=json.loads((r/'results/deep-actualprefix-r6993-preflight.json').read_text());assert sha(model/'config.json')==preq['config_SHA256']=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09'and sha(model/'model.safetensors.index.json')==preq['checkpoint_index_SHA256']=='7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a'
paramp=r/'results/deep-actualprefix-r6993-parameters.json';assert sha(paramp)==preq['source_index_SHA256'];params=[v for v in json.loads(paramp.read_text())['parameters']if v['key'].startswith('layers.0.')];expected={v['key']for v in params};oldweights={v['key']:v for v in json.loads((r/'results/deep-actualprefix-r6994-910374.587-CPUweights.json').read_text())if v['key']in expected};assert set(oldweights)==expected
refp=r/'results/deep-attention-r3509-910406.102.json';assert sha(refp)=='91c3d2891266bb9cdaea933bd5eefdd2e239103c1740b6bac475421c391d10c9';reference=json.loads(refp.read_text());natural=json.loads((r/'results'/f'deep-serving-cases-r2130-{tag}.json').read_text());assert [v['input_ids']for v in natural['cases']]==[v['input_ids']for v in reference['natural_input_provenance']['cases']];assert all(len(v['input_ids'])==512 for v in natural['cases']);frozen=next(c for c in reference['attention_F64_cases']if c['phase']=='natural-l0-r1')['generated_ids'];assert len(frozen)==8 and all(len(v)>=36 for v in frozen)
prompts=[natural['cases'][i%2]['input_ids']for i in range(32)];history=[frozen[i%8][:36]for i in range(32)];weights={};grouped={};weightproof=[];start=time.perf_counter()
for row in params:grouped.setdefault(row['file'],[]).append(row)
for fn,rows in sorted(grouped.items()):
 with safe_open(model/fn,framework='pt',device='cpu')as f:
  for row in rows:
   k=row['key'];t=f.get_tensor(k).contiguous();h=hashraw(memoryview(t.reshape(-1).view(torch.uint8).numpy()));assert list(t.shape)==row['shape']and h==oldweights[k]['CPU_tensor_SHA256'];weights[k]=t;weightproof.append({'key':k,'shape':list(t.shape),'dtype':str(t.dtype),'CPU_tensor_SHA256':h})
dtype_codes={torch.int8:0,torch.float32:1,torch.bfloat16:2,torch.int64:3}
assert set(weights)==expected
assert all(w.dtype in dtype_codes for w in weights.values()),'unsupported_checkpoint_dtype_before_owner_create'
unique=sorted({t for p in prompts for t in p}|{t for h in history for t in h});index={t:i for i,t in enumerate(unique)}
with safe_open(model/mapping['embed.weight'],framework='pt',device='cpu')as f:emb=torch.stack([f.get_slice('embed.weight')[t:t+1][0]for t in unique]).contiguous()
# Explicit OFF/ON/ON/OFF plus finalOFF restore, no model head or token generation.
arms=[0,1,1,0];barrier=threading.Barrier(8);cpu_set=sorted(os.sched_getaffinity(0));assert len(cpu_set)>=32;uid=ctypes.create_string_buffer(lib.deep_partial_id_size_r7003());assert lib.deep_partial_id_size_r7003()==136;check(lib.deep_partial_id_r7003(uid));rawroot=r/'results'/f'deep-direct-layer-r8714-{tag}-raw';rawroot.mkdir();report={'completed':False,'step':tag,'scope':'Actual native r7003 layer0, production weights/embedding/frozenprefix, TP8/B32/36teachersteps, C4/projectiontile16/quantization/mode5SUM unchanged; observerOFF controls versus2ON windows25/35. No copycandidate/protocol change/fullModel/performancegain. OrdinaryCPUtimestamps and stream eventspan are not puretransport or GPUready.','CPU_weight_seconds':time.perf_counter()-start,'weight_manifest_SHA256':hashraw(json.dumps(weightproof,sort_keys=True).encode()),'weights_count':len(weights),'providers':provider,'library_SHA256':sha(library),'build_SHA256':sha(bp),'arms':arms,'performance_adopted':False,'owners':[]}

def save():
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
save()
def owner(dev):
 ctx=None;records=[];baseline={};mask=None;expected_clock_sequence=0
 try:
  inherited=sorted(os.sched_getaffinity(0));assert inherited==cpu_set;torch.cuda.set_device(dev);ctx=lib.deep_partial_create_r7003(dev,json.dumps(cfg).encode(),uid);assert ctx,lib.deep_partial_error_r7003();names=json.loads(lib.deep_partial_keys_r7003(ctx));assert set(names)==expected
  for name in names:
   w=weights[name];shape=(Z*w.ndim)(*w.shape);check(lib.deep_partial_load_r7003(ctx,name.encode(),ptr(w),w.ndim,shape,dtype_codes[w.dtype]))
  check(lib.deep_partial_finish_r7003(ctx));check(lib.deep_output_clock_config_r8713(dev,0));mask=sorted(os.sched_getaffinity(0));assert len(mask)==4 and set(mask)<=set(inherited);barrier.wait(timeout=600)
  for arm,flag in enumerate(arms):
   # All owners complete prior arm before process-global env change.
   barrier.wait(timeout=180)
   if dev==0:os.environ['INFINILM_DEEP_OUTPUT_DIRECT_R7247']='0'
   barrier.wait(timeout=180);check(lib.deep_partial_reset_r7003(ctx));times=[];proofs=[]
   for step in range(36):
    for a in range(0,32,4 if step==0 else 32):
     rows=list(range(a,min(32,a+(4 if step==0 else 32))));chunks=[prompts[i]if step==0 else[history[i][step-1]]for i in rows];ids=sum(chunks,[]);past=[0 if step==0 else 512+step-1 for i in rows];pos=sum([list(range(p,p+len(v)))for p,v in zip(past,chunks)],[]);req=np.ascontiguousarray([[i,p,len(v)]for i,p,v in zip(rows,past,chunks)],dtype=np.uint64)
     x=emb[[index[t]for t in ids]].unsqueeze(1).expand(-1,4,-1).contiguous().to(dev);ti=torch.tensor(ids,dtype=torch.int64,device=dev);tp=torch.tensor(pos,dtype=torch.int64,device=dev)
     observe=bool(flag and step in[25,35]);check(lib.deep_output_clock_config_r8713(dev,int(observe)));oldtile=int(lib.deep_output_tile_calls_r4808(dev));barrier.wait(timeout=180);tick=time.perf_counter();check(lib.deep_partial_forward_r7003(ctx,len(ids),len(rows),ctypes.c_void_p(req.ctypes.data),ptr(x),ptr(ti),ptr(tp),None,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)));elapsed=time.perf_counter()-tick
     tile_delta=int(lib.deep_output_tile_calls_r4808(dev))-oldtile;clock=None;assert tile_delta==(1 if step else 0);
     if observe:
      slot=(ctypes.c_double*8)();check(lib.deep_output_clock_read_r8713(slot,8));clock=list(slot);expected_clock_sequence+=1;assert clock[6]==expected_clock_sequence and clock[7]==32 and all(clock[i]<=clock[i+1]for i in range(4))and clock[5]>0
     assert torch.cuda.mem_get_info(dev)[0]>=6*1024**3
     if step:times.append(elapsed)
     # Only selected attn_value/hidden windows are read outside CAPI timing.
     metas=json.loads(lib.deep_partial_stages_r7003(ctx));assert len(metas)==17;st=[]
     selected_metas=[(i,m)for i,m in enumerate(metas)if m['name']in['attn_value','hidden']]if step in[25,35]else[]
     for si,meta in selected_metas:
      buf=(ctypes.c_ubyte*meta['bytes'])();check(lib.deep_partial_read_r7003(ctx,si,buf,meta['bytes']));raw=bytes(buf);name=meta['name'];dtype=meta['dtype'];arr=np.frombuffer(raw,{'BF16':np.uint16,'F32':np.float32,'I32':np.int32,'I64':np.int64}[dtype]);values=(arr.astype(np.uint32)<<16).view(np.float32)if dtype=='BF16'else arr
      assert np.isfinite(values).all();key=(step,a,name);h=hashraw(raw)
      if arm==0:baseline[key]=raw if step else h
      old=baseline[key];exact=h==(old if isinstance(old,str)else hashraw(old));numeric=True;maxabs=0.
      if step and not exact:
       oldarr=np.frombuffer(old,{'BF16':np.uint16,'F32':np.float32,'I32':np.int32,'I64':np.int64}[dtype]);oldvalues=(oldarr.astype(np.uint32)<<16).view(np.float32)if dtype=='BF16'else oldarr
       if dtype in ['I32','I64']:numeric=False
       else:maxabs=float(np.max(np.abs(values-oldvalues)));numeric=bool(np.allclose(values,oldvalues,rtol=1e-3,atol=1e-3))
      if not step:numeric=exact
      assert numeric,('component_stage_numeric_regression',dev,arm,step,name,maxabs)
      # Byte equality diagnostic only; mathematical path unchanged.
      z={'name':name,'shape':meta['shape'],'dtype':dtype,'bytes':len(raw),'SHA256':h,'OFF0_exact':exact,'numeric_guard':numeric,'maxabs':maxabs}
      st.append(z)
     proofs.append({'step':step,'chunk_begin':a,'rows':rows,'selected_output_capture':bool(selected_metas),'tile_delta':tile_delta,'observer_clock':clock,'CAPI_wall_seconds':elapsed,'stages':st});barrier.wait(timeout=180)
   records.append({'arm':arm,'observer':flag,'whole_layer_decode_seconds':sum(times),'decode_forward_count':len(times),'forwards':proofs});barrier.wait(timeout=180)
  return {'dev':dev,'affinity':mask,'arms':records,'normal_owner_closed':True}
 except BaseException:
  barrier.abort();raise
 finally:
  if ctx:check(lib.deep_partial_destroy_r7003(ctx))
  check(lib.deep_output_clock_config_r8713(dev,2))
  torch.cuda.synchronize(dev)
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:report['owners']=list(pool.map(owner,range(8)))
 masks=[set(o['affinity'])for o in report['owners']];assert len(masks)==8 and all(len(v)==4 for v in masks)and len(set.union(*masks))==32
 os.environ['INFINILM_DEEP_OUTPUT_DIRECT_R7247']='0';report['owner_masks_disjoint_verified']=True;report['completed']=True;report['projection_only_time']='NOT_MEASURED';report['clock_fields']=['CPU_forward_begin','CPU_local_projection_enqueue_end','CPU_enter_SUM','CPU_return_SUM_enqueue','CPU_after_original_drain','GPU_event_span_s','owner_sequence','tokens'];report['scope_clock']='SamehostCPUenclosing/queueevents only; CPUenqueue not GPUready; eventspan includesarrival/hostenqueue/CCL, not puretransport. ObserverOFF arms0/3 control, ON arms1/2 onlysteps25/35; no math/copy/protocol changes.';report['no_serving_TPS']=True
 # Max-owner trajectory wall summarizes slowest owner, not sum of TP rank times.
 report['arm_max_owner_seconds']=[max(o['arms'][i]['whole_layer_decode_seconds']for o in report['owners'])for i in range(4)]
 report['ABBA_wholelayer_ratio']=sum(report['arm_max_owner_seconds'][i]for i in [0,3])/sum(report['arm_max_owner_seconds'][i]for i in [1,2]);report['repeat_noise_note']='ObserverOFF complete layer control arms0/3; observerON arms1/2 onlysteps25/35. This is observation overhead, not an optimization gain.'
except BaseException as e:report['error']=repr(e);raise
finally:report['elapsed_seconds']=time.perf_counter()-start;save()
print(json.dumps({'completed':report['completed'],'path':str(path),'ABBA_wholelayer_ratio':report.get('ABBA_wholelayer_ratio')}),flush=True)
