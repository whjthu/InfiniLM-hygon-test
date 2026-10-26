"""One actual TP8 checkpoint CSA layer; synthetic normalized inputs, not Model/task throughput."""
import pathlib,json,hashlib,ctypes,os,time,threading,concurrent.futures,traceback,statistics,contextlib
import numpy as np
import torch
from safetensors import safe_open
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=r/'results'/('deep-c4-smallt-r9252b-'+tag+'.json');assert not out.exists();manifest=json.loads((inc/'manifest.json').read_bytes());cp=r/'candidates/deep-c4-smallt-r9252/CPU-build.json';assert sha(cp)==manifest['CPU_receipt_SHA256'];cpu=json.loads(cp.read_bytes());assert cpu['completed']and cpu['library_SHA256']==manifest['DSO_SHA256']==sha(cpu['library']);assert len(cpu['reused_objects_SHA256'])==48 and all(sha(f)==h for f,h in cpu['reused_objects_SHA256'].items());assert all(sha(f)==v['SHA256']for f,v in cpu['actual_DTK_bindings'].items());torch.set_num_threads(1);assert len(os.sched_getaffinity(0))>=32 and torch.cuda.device_count()==8;torch.cuda.set_device(0);lib=ctypes.CDLL(cpu['library']);ptr=lambda t:ctypes.c_void_p(t.data_ptr());lib.deep_csa_fixture_error_r9242.restype=ctypes.c_char_p
for n,args,result in [('id_size',[],ctypes.c_size_t),('id',[ctypes.c_void_p],ctypes.c_int),('create',[ctypes.c_int,ctypes.c_char_p,ctypes.c_void_p],ctypes.c_void_p),('keys',[ctypes.c_void_p],ctypes.c_char_p),('load',[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int],ctypes.c_int),('forward',[ctypes.c_void_p]+[ctypes.c_size_t]*2+[ctypes.c_void_p]*6,ctypes.c_int),*[(v,[ctypes.c_void_p],ctypes.c_int)for v in ['finish','reset','destroy']],('shutdown',[],ctypes.c_int)]:
 f=getattr(lib,'deep_csa_fixture_'+n+'_r9242');f.argtypes=args;f.restype=result
lib.modern_deep_c4_smallt_mode_r9252.argtypes=[ctypes.c_int];lib.modern_deep_c4_smallt_mode_r9252.restype=ctypes.c_int
lib.modern_deep_c4_smallt_stats_r9252.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.modern_deep_c4_smallt_stats_r9252.restype=ctypes.c_int
lib.deep_csa_host_profile_mode_r9246.argtypes=[ctypes.c_int];lib.deep_csa_host_profile_mode_r9246.restype=ctypes.c_int
lib.deep_csa_host_profile_read_r9246.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.deep_csa_host_profile_read_r9246.restype=ctypes.c_int
for n in ['modern_deep_precision_policy_r3443','modern_deep_allselected_set_r813','deep_csa_readback_mode_r9242','deep_latent_prefill_mode_r8933','deep_reference_scratch_mode_r8953','deep_reference_packed_batch_mode_r8996b','deep_sink_broadcast_mode_r9040','deep_scalar_cast_mode_r9054','deep_values_reuse_mode_r9083','deep_mode1_sink_r9207','modern_deep_c4_tile_policy_r5920']:
 f=getattr(lib,n);f.argtypes=[ctypes.c_int];f.restype=ctypes.c_int
for n in ['modern_deep_latent_fused_calls_r681','modern_deep_latent_fallback_calls_r681','modern_deep_FP32_calls_r3597']:
 f=getattr(lib,n);f.argtypes=[ctypes.c_int];f.restype=ctypes.c_uint64
lib.modern_deep_allselected_calls_r813.argtypes=[ctypes.c_int]*2;lib.modern_deep_allselected_calls_r813.restype=ctypes.c_uint64
for n in ['deep_csa_readback_counts_r9242','modern_deep_precision_stats_r3443']:
 f=getattr(lib,n);f.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];f.restype=ctypes.c_int

def check(rc):
 if rc:raise RuntimeError((rc,lib.deep_csa_fixture_error_r9242().decode()))
for n,v in [('modern_deep_c4_smallt_mode_r9252',0),('deep_csa_host_profile_mode_r9246',0),('modern_deep_precision_policy_r3443',5),('modern_deep_allselected_set_r813',1),('deep_csa_readback_mode_r9242',0),('deep_latent_prefill_mode_r8933',0),('deep_reference_scratch_mode_r8953',1),('deep_reference_packed_batch_mode_r8996b',1),('deep_sink_broadcast_mode_r9040',1),('deep_scalar_cast_mode_r9054',1),('deep_values_reuse_mode_r9083',0),('deep_mode1_sink_r9207',1),('modern_deep_c4_tile_policy_r5920',16)]:check(getattr(lib,n)(v))
model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');assert sha(model/'config.json')=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09';cfg=json.loads((model/'config.json').read_bytes());assert cfg['compress_ratios'][16]==4 and cfg['num_hidden_layers']==43;mapping=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];weights={};headers={};stack=contextlib.ExitStack();loadbegan=time.perf_counter();groups={}
for k,fn in mapping.items():
 if k.startswith('layers.16.attn.'):groups.setdefault(fn,[]).append(k)
for fn,names in groups.items():
 sf=model/fn
 with sf.open('rb')as f:n=int.from_bytes(f.read(8),'little');raw=f.read(n)
 headers[fn]={'size':sf.stat().st_size,'header_SHA256':hashlib.sha256(raw).hexdigest(),'full_payload_rehashed':False};f=stack.enter_context(safe_open(sf,framework='pt',device='cpu'))
 for k in names:
  if k not in f.keys():continue # Preserve proven loader: index may contain nonphysical optional scale alias.
  w=f.get_tensor(k);assert w.is_contiguous();weights[k]=w
assert weights
uid=ctypes.create_string_buffer(lib.deep_csa_fixture_id_size_r9242());check(lib.deep_csa_fixture_id_r9242(uid));barrier=threading.Barrier(8);lock=threading.Lock();began=time.perf_counter();report={'completed':False,'normal':False,'step':tag,'DSO_SHA256':cpu['library_SHA256'],'CPU_receipt_SHA256':sha(cp),'scope':'ONE actualcheckpointCSA16/TP8; defaultOFFC4T1..7decodeallselectedguard;keys/compressor/commit/revisionpreserved;synthetichidden/currentpasts;HOSTinstrument/readbackOFF;T8B32legacycontrols;notModel/taskquality.','checkpoint':{'path':str(model),'config_SHA256':sha(model/'config.json'),'index_SHA256':sha(model/'model.safetensors.index.json'),'headers':headers,'CPU_mmap_registered_seconds':time.perf_counter()-loadbegan,'full_RAM_residency_claim':False},'owners':[{'device':d,'measurements':[],'normal_owner_closed':False}for d in range(8)],'capacity_component_only':{'rows':32,'max_positions':4096,'compressed_capacity':1024},'plan':'T1T2T4T7+T8B32controls firstseparate/2warm4pairs;continuoussame-state512to513mixedthreshold/EOSretirement+causalguards+zeroVisibleQA;directIndexerabort/staleforeigncommitNOTCOVERED;HOSTinstrumentOFF/readbackOFF;resetoutsideperfclock/allocationpoolnottrimmed'}
def save():
 with lock:
  t=out.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2));t.replace(out)
def ct(name,dev,n):
 a=(ctypes.c_uint64*n)();check(getattr(lib,name)(dev,a,n));return list(a)
def host_profile(dev):
 w=(ctypes.c_double*14)();c=(ctypes.c_double*14)();m=(ctypes.c_uint64*6)();v=(ctypes.c_uint64*13)();check(lib.deep_csa_host_profile_read_r9246(dev,w,c,m,v,14));return {'wall_seconds':list(w),'thread_CPU_seconds':list(c),'meta':list(m),'visits':list(v)}
def snapshot(dev):return {'smallt':ct('modern_deep_c4_smallt_stats_r9252',dev,8),'readback':ct('deep_csa_readback_counts_r9242',dev,8),'precision':ct('modern_deep_precision_stats_r3443',dev,6),'allselected':int(lib.modern_deep_allselected_calls_r813(dev,1)),'fullindexer':int(lib.modern_deep_allselected_calls_r813(dev,0)),'fused':int(lib.modern_deep_latent_fused_calls_r681(dev)),'reference':int(lib.modern_deep_latent_fallback_calls_r681(dev)),'FP32attention':int(lib.modern_deep_FP32_calls_r3597(dev))}
def delta(a,b):return {k:([y-x for x,y in zip(v,b[k])]if isinstance(v,list)else b[k]-v)for k,v in a.items()}
past32=[413,433,458,377,416,387,423,436,436,391,414,388,415,389,390,406,445,418,427,396,396,411,414,454,392,379,421,383,418,451,383,400]
tailrows=[2,7,16,22,23,26,29];tailpast=[570,548,557,526,566,533,563]
specs=[{'name':'realT'+str(T),'rows':tailrows[:T],'past':tailpast[:T],'kind':'performance'}for T in[1,2,4,7]]+[{'name':'legacyT8','rows':list(range(8)),'past':past32[:8],'kind':'performance'},{'name':'legacyB32','rows':list(range(32)),'past':past32,'kind':'performance'},{'name':'continuous_threshold_EOS7','rows':tailrows,'past':[2043,2044,2045,2046,2043,2044,2045],'reserve_tokens':20,'steps':10,'retire_after':5,'kind':'sequence'},{'name':'zero_visible_T1','rows':[27],'past':[1],'reserve_tokens':8,'steps':4,'kind':'sequence'},{'name':'zero_visible_T7','rows':tailrows,'past':[1]*7,'reserve_tokens':8,'steps':4,'kind':'sequence'}];report['fixtures']=specs;report['host_bucket_names']=['indexer_prepare_inclusive','selection_D2H_inclusive_wait','selection_validation_HOST','bank_tensor_alloc_copy_submit_inclusive','descriptors_HOST','explicit_original_wait_drain','prepare_projection_norm_RoPE','compressor_prepare_inclusive','latent_inclusive','inverse_RoPE','outputprojection_collective_inclusive','commitcache_indexer_compressor_inclusive','residual','wholeCSA_forward_total'];report['timer_boundaries']='ExclusiveHOSTphaseintervals,notpureGPUbusy. Helperinternalwaitsremaininsideinclusivebucket;proofcopy/CAPIstagingoutsideFrameincludedwholeCAPIwall. Noallocationpoolclear,reset/prefixoutsideclock.'
# Shared read-only CPU source per case/row; deterministic same normalized tensors on all TP ranks.
source={}
for si,spec in enumerate(specs):
 for row,past in zip(spec['rows'],spec['past']):
  gen=torch.Generator().manual_seed(924200+si*100+row);x=torch.randn((past+spec.get('reserve_tokens',2),4096),generator=gen);x=(x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()).to(torch.bfloat16);source[(si,row)]=x
save()
def owner(dev):
 torch.cuda.set_device(dev);rec=report['owners'][dev];ctx=None;baseline={}
 try:
  # Runtime applies its original width4 hook; don't narrow thread before create.
  ctx=lib.deep_csa_fixture_create_r9242(dev,json.dumps(cfg).encode(),uid);assert ctx,lib.deep_csa_fixture_error_r9242().decode();rec['owner_tid']=threading.get_native_id();rec['CPU_affinity_after_create']=sorted(os.sched_getaffinity(0));assert len(rec['CPU_affinity_after_create'])==4;names=json.loads(lib.deep_csa_fixture_keys_r9242(ctx));assert set(names)<=set(weights);rec['loaded_keys']=names
  for name in names:
   w=weights[name];shape=(ctypes.c_size_t*w.ndim)(*w.shape);check(lib.deep_csa_fixture_load_r9242(ctx,name.encode(),ptr(w),w.ndim,shape,{torch.int8:0,torch.float32:1,torch.bfloat16:2,torch.int64:3}[w.dtype]))
  check(lib.deep_csa_fixture_finish_r9242(ctx));barrier.wait(240)
  def forward(si,entries,measured=False):
   spec=specs[si];xx=[];pp=[];req=[]
   for row,past,count in entries:xx.append(source[(si,row)][past:past+count]);pp+=list(range(past,past+count));req.append([row,past,count])
   x=torch.cat(xx).to(dev);pos=torch.tensor(pp,dtype=torch.int64,device=dev);ids=torch.zeros(len(pp),dtype=torch.int64,device=dev);req=np.ascontiguousarray(req,np.uint64);before_x=x.clone()if measured else None;before_pos=pos.clone()if measured else None;back=torch.full((len(pp)*4096+128,),42.,device=dev,dtype=torch.bfloat16)if measured else None;proof=back[:len(pp)*4096].view(len(pp),4096)if measured else None;stream=torch.cuda.current_stream();torch.cuda.synchronize(dev)
   if measured:barrier.wait(180);before=snapshot(dev);profile_before=host_profile(dev)
   tick=time.perf_counter();rc=lib.deep_csa_fixture_forward_r9242(ctx,len(pp),len(entries),ctypes.c_void_p(req.ctypes.data),ptr(x),ptr(ids),ptr(pos),ptr(proof)if measured else None,ctypes.c_void_p(stream.cuda_stream));wall=time.perf_counter()-tick;check(rc)
   if not measured:return
   after=snapshot(dev);profile_after=host_profile(dev);a=proof.float();finite=bool(torch.isfinite(a).all());canary=bool((back[-128:]==42).all());readonly=torch.equal(x,before_x)and torch.equal(pos,before_pos);assert finite and canary and readonly;numeric=None
   key=(si,tuple(entries))
   if key not in baseline:baseline[key]=a.clone()
   else:
    ref=baseline[key];numeric={'maxabs':float((a-ref).abs().max()),'relative_RMS':float((a-ref).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-9))};assert numeric['relative_RMS']<=.02,numeric
   profile=None
   if profile_after['meta'][1]>profile_before['meta'][1]:
    assert profile_after['meta'][1]==profile_before['meta'][1]+1 and profile_after['meta'][2:4]==[len(pp),len(entries)] and profile_after['meta'][5]==1;profile=profile_after
    assert abs(sum(profile['wall_seconds'][:-1])-profile['wall_seconds'][-1])<1e-7 and abs(sum(profile['thread_CPU_seconds'][:-1])-profile['thread_CPU_seconds'][-1])<1e-7
    assert profile['wall_seconds'][-1]<=wall+.0001
   return {'smallt_before':before['smallt'],'smallt_after':after['smallt'],'host_profile':profile,'wall_seconds':wall,'counts':delta(before,after),'finite':finite,'readonly':readonly,'canary':canary,'numerical':numeric,'driver_free_total':list(torch.cuda.mem_get_info())}
  def select(mode,rows):
   barrier.wait(180)
   if dev==0:check(lib.modern_deep_c4_smallt_mode_r9252(mode));os.environ['INFINILM_DEEP_OUTPUT_TILE_R4808']='16'if len(rows)==32 else'0'
   barrier.wait(180)
  def prefix(si):
   spec=specs[si];barrier.wait(180)
   if dev==0:check(lib.modern_deep_c4_smallt_mode_r9252(0));check(lib.deep_csa_host_profile_mode_r9246(0));check(lib.deep_csa_readback_mode_r9242(0));os.environ['INFINILM_DEEP_OUTPUT_TILE_R4808']='0'
   barrier.wait(180);check(lib.deep_csa_fixture_reset_r9242(ctx))
   for start in range(0,max(spec['past']),512):
    entries=[(row,start,min(512,past-start))for row,past in zip(spec['rows'],spec['past'])if past>start];batch=[];size=0
    for e in entries:
     if size+e[2]>2048:forward(si,batch);batch=[];size=0
     batch.append(e);size+=e[2]
    if batch:forward(si,batch)
   barrier.wait(180)
  def verify(m,entries,mode):
   T=len(entries);visible=[(past+1)//4 for row,past,count in entries];legacy=T>=8 and T<=64 and max(visible)<=512;short=T<=7 and all(past>0 for row,past,count in entries);eligible=short and max(visible)<=512;fast=legacy or bool(mode and eligible);c=m['counts'];assert m['host_profile']is None;assert c['allselected']==int(fast)and c['fullindexer']==int(not fast)and c['FP32attention']==int(T>1) and c['fused']==int(T>=8)and c['reference']==int(T<8),c
   assert c['smallt'][:4]==[int(short),int(eligible),int(short and fast),int(short and not fast)],c
   assert m['smallt_after'][4]==m['smallt_before'][4]+1 and m['smallt_after'][5:]==[T,T,int(eligible)],m
   if T>=2:assert c['readback'][:5]==[1,0,T,0,0],c
   else:assert c['readback']==[0]*8,c
   assert(c['precision']==[0]*6 if T==1 else c['precision'][0]==(1 if fast else T+1)),c
  def invalid_guard(si,entries,label):
   xx=[];positions=[]
   for row,past,count in entries:xx.append(source[(si,row)][past:past+count]);positions+=list(range(past,past+count))
   x=torch.cat(xx).to(dev);pos=torch.tensor(positions,dtype=torch.int64,device=dev);ids=torch.zeros(len(positions),dtype=torch.int64,device=dev);req=np.ascontiguousarray(entries,np.uint64);torch.cuda.synchronize(dev);barrier.wait(180);before=snapshot(dev);rc=lib.deep_csa_fixture_forward_r9242(ctx,len(positions),len(entries),ctypes.c_void_p(req.ctypes.data),ptr(x),ptr(ids),ptr(pos),None,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream));after=snapshot(dev);assert rc and before==after,('invalid_guard_state_mutated',label,rc,before,after);rec.setdefault('guard_rejections',[]).append({'case':specs[si]['name'],'label':label,'error':lib.deep_csa_fixture_error_r9242().decode()[:240],'successful_prepare_state_unchanged':True});barrier.wait(180)
  for si,spec in enumerate(specs):
   if spec['kind']=='performance':
    for phase,mode in [('first',1),('warm',0),('warm',1)]+[('measure',v)for v in[0,1,1,0,0,1,1,0]]+[('restore',0)]:
     if dev==0:report['active']={'fixture':spec['name'],'phase':phase,'arm':mode,'stage':'prefix'};save()
     prefix(si);select(mode,spec['rows']);entries=[(row,past,1)for row,past in zip(spec['rows'],spec['past'])];m=forward(si,entries,True);verify(m,entries,mode);m.update(fixture=spec['name'],phase=phase,arm=mode,rows=spec['rows'],past=spec['past'],outputtile=16 if len(entries)==32 else 0);rec['measurements'].append(m);barrier.wait(180)
     if dev==0:save()
   else:
    for mode in[0,1]:
     if dev==0:report['active']={'fixture':spec['name'],'phase':'continuous_QA','arm':mode,'stage':'prefix_once'};save()
     prefix(si);previous=None
     for step in range(spec['steps']):
      rows=spec['rows'][1:]if spec.get('retire_after')is not None and step>=spec['retire_after']else spec['rows'];initial=dict(zip(spec['rows'],spec['past']));entries=[(row,initial[row]+step,1)for row in rows];select(mode,rows);m=forward(si,entries,True);verify(m,entries,mode)
      if previous is not None:assert m['smallt_after'][4]==previous+1,('continuous_revision',m,previous)
      previous=m['smallt_after'][4];m.update(fixture=spec['name'],phase='continuous_QA',arm=mode,step=step,rows=rows,past=[x[1]for x in entries],no_reset_since_prefix=True);rec['measurements'].append(m);barrier.wait(180)
      if step==0:
       invalid_guard(si,entries,'stale_past_replay_after_commit')
       if len(rows)>=2:
        row=rows[0];past=initial[row]+1;invalid_guard(si,[(row,past,1),(row,past,1)],'duplicate_live_row')
      if spec.get('retire_after')==step:
       row=spec['rows'][0];invalid_guard(si,[(row,initial[row]+step+1,1)],'retired_EOS_row_cannot_skip_past')
      if dev==0:save()
  barrier.wait(180)
  if dev==0:check(lib.modern_deep_c4_smallt_mode_r9252(0));check(lib.deep_csa_readback_mode_r9242(0));check(lib.deep_csa_host_profile_mode_r9246(0))
  barrier.wait(180)

 except BaseException as e:
  rec.update(error=repr(e),traceback=traceback.format_exc());save();barrier.abort();raise
 finally:
  if ctx:check(lib.deep_csa_fixture_destroy_r9242(ctx));ctx=None
  check(lib.deep_csa_fixture_shutdown_r9242());torch.cuda.synchronize(dev);rec['normal_owner_closed']=True;save()
 return rec
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(owner,range(8)))
 report['normal']=all(o['normal_owner_closed']for o in report['owners']);report['completed']=True;report['active']=None
 for spec in specs[:6]:
  values=[m for m in report['owners'][0]['measurements']if m['fixture']==spec['name']and m['phase']=='measure'];paired=[]
  for ix,m in enumerate(values):
   vals=[[v for v in o['measurements']if v['fixture']==spec['name']and v['phase']=='measure'][ix]['wall_seconds']for o in report['owners']];paired.append({'arm':m['arm'],'matched_iteration_max_owner_seconds':max(vals)})
  spec['performance']={'matched_iterations':paired,'OFF_median_seconds':statistics.median(v['matched_iteration_max_owner_seconds']for v in paired if v['arm']==0),'ON_median_seconds':statistics.median(v['matched_iteration_max_owner_seconds']for v in paired if v['arm']==1)}
  spec['performance']['wall_change_pct']=(spec['performance']['ON_median_seconds']/spec['performance']['OFF_median_seconds']-1)*100
except BaseException as e:report['error']=repr(e);save();raise
finally:report['elapsed_seconds']=time.perf_counter()-began;save();stack.close()
print(json.dumps({'completed':report['completed'],'normal':report['normal'],'output':str(out)}))
