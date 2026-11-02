"""Whole TP8 C4 prepare/commit; synthetic inputs, actual checkpoint weights.

No default launch: a frozen GPU manifest, source claim and bounded controller
must exist first. Raw weights/features/proofs remain in process, never JSON.
"""
from pathlib import Path
import ctypes as C,os,json,hashlib,time,threading,concurrent.futures,traceback,statistics
import numpy as np
R=Path('/refactor');S=Path(__file__).resolve().parent
P=C.c_void_p;Z=C.c_size_t;I=C.c_int
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def plan():
 arms=[0,1,1,0]*5
 return [dict(past=p,phase=kind,mode=m)for p in [8496,8575]
         for kind,m in [('baseline_QA',0),('private_fallback_QA',2),('candidate_QA',1),('warm',0),('warm',1)]+[('measurement',m)for m in arms]+[('restore',0)]]

def configure(lib):
 f=lambda n:getattr(lib,'deep_c4_fixture_'+n+'_r359')
 f('error').restype=C.c_char_p
 for n,args,result in [
  ('id_size',[],Z),('id',[P],I),('create',[I,C.c_char_p,P],P),('keys',[P],C.c_char_p),
  ('load',[P,C.c_char_p,P,Z,C.POINTER(Z),I],I),('finish',[P],I),('reset',[P,I],I),
  ('forward',[P,Z,Z,P,P,P,P],I),('proof_info',[P,I,C.POINTER(Z),C.POINTER(Z),C.POINTER(I)],I),
  ('proof_read',[P,I,P,Z],I),('stats',[P,C.POINTER(C.c_double),C.POINTER(Z)],I),('destroy',[P],I),('shutdown',[],I)]:
  f(n).argtypes=args;f(n).restype=result
 return f

def proof(f,check,ctx,which):
 shape=(Z*5)();size=Z();dtype=I();check(f('proof_info')(ctx,which,shape,C.byref(size),C.byref(dtype)))
 if not size.value:return None
 data=np.empty(size.value,np.uint8);check(f('proof_read')(ctx,which,P(data.ctypes.data),size.value))
 return dict(raw=data,shape=tuple(shape[i+1]for i in range(shape[0])),dtype=dtype.value)

def numeric_all(proofs,oracle,baseline):
 """CPU gather comes from original raw page layout, never candidate gather."""
 total=None
 for q in proofs:
  before=q['past']//4;after=(q['past']+1)//4
  raw=q['fields'][6]['raw'].reshape(q['fields'][6]['shape'])
  keys,scale=oracle.gather_paged(raw,np.arange(after),np.ones(after,np.uint8))
  query=q['fields'][0]['raw'].reshape(8,128)
  weights=q['fields'][1]['raw'].view(np.float32).reshape(8)
  local=oracle.score_cpu(query,weights,keys,scale.reshape(-1),after)
  total=local if total is None else np.add(total,local,dtype=np.float32)
 metrics=[]
 for q in proofs:
  scores=q['fields'][4]['raw'].view(np.float32).reshape(-1)
  ids=q['fields'][5]['raw'].view(np.int64).reshape(-1)
  assert len(scores)==len(total)and len(ids)==512 and len(set(ids.tolist()))==512
  assert np.isfinite(scores).all()and np.all((ids>=0)&(ids<len(scores)))
  diff=scores.astype(np.float64)-total.astype(np.float64)
  nrmse=float(np.linalg.norm(diff)/(np.linalg.norm(total.astype(np.float64))+1e-30))
  assert nrmse<2e-4,(q['rank'],nrmse)
  cutoff=float(np.partition(scores,-512)[-512]);tol=max(1e-5,float(np.max(np.abs(scores)))*2e-5)
  assert np.min(scores[ids])>=cutoff-tol
  b=baseline[q['rank']]if baseline else None
  item=dict(rank=q['rank'],CPU_FP8_score_NRMSE=nrmse,CPU_maxabs=float(np.max(np.abs(diff))),finite=True,TopK512_unique_valid=True)
  if b:
   # Feature generation/cache/committed compressor are unchanged by score fusion.
   for field in [0,1,2,3,7]:
    x,y=q['fields'][field],b['fields'][field]
    assert (x is None)==(y is None)
    if x is not None:assert x['shape']==y['shape']and np.array_equal(x['raw'],y['raw']),(field,q['rank'])
   old=b['fields'][4]['raw'].view(np.float32).reshape(-1)
   item.update(original_score_maxabs=float(np.max(np.abs(scores-old))),TopK_overlap=len(set(ids.tolist())&set(b['fields'][5]['raw'].view(np.int64).tolist())))
  metrics.append(item)
 return metrics

def main():
 import torch,importlib.util,contextlib
 from safetensors import safe_open
 assert os.environ.get('DEEP359_GPU_GO')=='FROZEN_CLAIM'
 manifest=json.loads((S/'GPU_MANIFEST.json').read_bytes())
 assert all(sha(S/n)==h for n,h in manifest['source_SHA256'].items())
 receipt=Path(manifest['CPU_receipt']);assert sha(receipt)==manifest['CPU_receipt_SHA256']
 cpu=json.loads(receipt.read_bytes());assert cpu['completed']and sha(cpu['library'])==cpu['library_SHA256']==manifest['DSO_SHA256']
 assert cpu['library_SHA256']=='a4e6208eef89bd63045c01bcab5ccc358e6a36cd6ea034aa286b2934ba82d91b'
 assert len(os.sched_getaffinity(0))>=32
 deadline=float(os.environ['DEEP359_ABSOLUTE_DEADLINE']);began=time.perf_counter()
 def remaining():
  assert time.time()<deadline-10,'Original GPU deadline exhausted'
 torch.set_num_threads(1);remaining();assert torch.cuda.device_count()==8
 cuda=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so');cuda.cudaDeviceGetPCIBusId.argtypes=[C.c_char_p,I,I]
 identities=[]
 for d in range(8):
  props=torch.cuda.get_device_properties(d);assert 'gfx936'in props.gcnArchName
  bus=C.create_string_buffer(64);assert cuda.cudaDeviceGetPCIBusId(bus,64,d)==0
  identities.append(dict(device=d,PCI=bus.value.decode(),arch=props.gcnArchName,name=props.name))
 assert len({r['PCI']for r in identities})==8
 source=R/'incoming/Deep-C4-paged-score-r359-source-cpu-v2';assert sha(source/'SOURCE_MANIFEST.json')=='38f8a95cfa8c6c7669ea43f66d1cb7d9507182a167feae23e2e93e1c762fc4ae'
 mm=json.loads((source/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(source/n)==h for n,h in mm['files_SHA256'].items())
 spec=importlib.util.spec_from_file_location('paged_score_oracle',source/'reference.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
 model=Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8')
 assert sha(model/'config.json')=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09'
 cfg=json.loads((model/'config.json').read_bytes());mapping=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map']
 assert cfg['compress_ratios'][16]==4
 weights={};headers={};loadstart=time.perf_counter()
 groups={}
 for k,fn in mapping.items():
  if k.startswith('layers.16.attn.indexer.'):groups.setdefault(fn,[]).append(k)
 with contextlib.ExitStack()as stack:
  for fn,names in groups.items():
   remaining();sf=model/fn
   with sf.open('rb')as f:n=int.from_bytes(f.read(8),'little');header=f.read(n)
   headers[fn]=dict(size=sf.stat().st_size,header_SHA256=hashlib.sha256(header).hexdigest(),full_payload_rehashed=False)
   reader=stack.enter_context(safe_open(sf,framework='pt',device='cpu'))
   for k in names:
    if k in reader.keys():weights[k]=reader.get_tensor(k).contiguous()
  assert weights
  seed=torch.Generator().manual_seed(3598496)
  # Fixed full unpadded input, same values for all TP ranks and both variants.
  x=torch.randn((8577,4096),generator=seed).to(torch.bfloat16)
  q=torch.randn((8577,1024),generator=seed).to(torch.bfloat16)
  lib=C.CDLL(cpu['library']);f=configure(lib)
  def check(rc):
   if rc:raise RuntimeError(f('error')().decode())
  uid=C.create_string_buffer(f('id_size')());check(f('id')(uid))
  barrier=threading.Barrier(8);lock=threading.Lock();shared=[None]*8;baselines={};reference_metrics={}
  step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('Deep359-whole-C4-'+step+'.json');assert not out.exists()
  report=dict(completed=False,normal=False,actual_step=step,hardware=identities,plan=plan(),owners=[dict(rank=d,calls=[],normal_close=False)for d in range(8)],CPU_receipt_SHA256=sha(receipt),DSO_SHA256=sha(cpu['library']),checkpoint=dict(config_SHA256=sha(model/'config.json'),index_SHA256=sha(model/'model.safetensors.index.json'),headers=headers),input_kind='Synthetic BF16 hidden/q_lora, real checkpoint weights; not Model activations or throughput',checkpoint_mmap_seconds=time.perf_counter()-loadstart,timing_scope='CAPI wall measures complete prepare+commit, input H2D/clone/validation and prefix reset/reseed excluded and separately recorded; measured checked envelope includes them; Torch peak excludes InfiniRT allocator, no whole-component peak claim')
  def save():
   with lock:
    temp=out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out)
  def stats(ctx):
   times=(C.c_double*2)();hits=Z();check(f('stats')(ctx,times,C.byref(hits)));return list(times),hits.value
  def owner(rank):
   ctx=None;r=report['owners'][rank];initial=sorted(os.sched_getaffinity(0));assigned=initial[rank*4:rank*4+4]
   try:
    remaining();torch.cuda.set_device(rank);torch.cuda.reset_peak_memory_stats(rank)
    ctx=f('create')(rank,json.dumps(cfg).encode(),uid);assert ctx,f('error')().decode()
    r.update(TID=threading.get_native_id(),initial_affinity=initial,expected_affinity=assigned)
    names=json.loads(f('keys')(ctx));assert set(names)<=set(weights);r['loaded_weight_keys']=names
    for name in names:
     remaining();w=weights[name];shape=(Z*w.ndim)(*w.shape);dtype={torch.int8:0,torch.float32:1,torch.bfloat16:2}[w.dtype]
     check(f('load')(ctx,name.encode(),P(w.data_ptr()),w.ndim,shape,dtype))
    check(f('finish')(ctx));barrier.wait(120)
    def forward(past,count):
     remaining();xx=x[past:past+count].to(rank);qq=q[past:past+count].to(rank);pos=torch.arange(past,past+count,dtype=torch.int64,device=rank)
     beforex=xx.clone();beforeq=qq.clone();torch.cuda.synchronize(rank)
     tick=time.perf_counter();check(f('forward')(ctx,past,count,P(xx.data_ptr()),P(qq.data_ptr()),P(pos.data_ptr()),P(torch.cuda.current_stream(rank).cuda_stream)));wall=time.perf_counter()-tick
     assert torch.equal(xx,beforex)and torch.equal(qq,beforeq)
     return wall
    for trial,item in enumerate(plan()):
     barrier.wait(120);remaining();check(f('reset')(ctx,item['mode']))
     prefix_start=time.perf_counter()
     for p in range(0,item['past'],1024):forward(p,min(1024,item['past']-p))
     prefix_seconds=time.perf_counter()-prefix_start
     # First original submit must select this exact rank's four CPU cores.
     r['affinity_after_submit']=sorted(os.sched_getaffinity(0));assert r['affinity_after_submit']==assigned
     before=stats(ctx)[1];barrier.wait(120);begin=time.perf_counter();wall=forward(item['past'],1);end=time.perf_counter();times,hits=stats(ctx)
     assert hits-before==int(item['mode']==1)
     fields={k:proof(f,check,ctx,k)for k in range(8)}
     shared[rank]=dict(rank=rank,past=item['past'],fields=fields)
     barrier.wait(120)
     if rank==0:
      key=item['past'];base=baselines.get(key)
      reference_metrics[trial]=numeric_all(shared,oracle,base)
      if base is None:baselines[key]=list(shared)
     barrier.wait(120)
     rec=dict(trial=trial,**item,prefix_reseed_wall_s=prefix_seconds,CAPI_wall_s=wall,measurement_begin_s=begin,measurement_end_s=end,H2D_and_checked_envelope_s=end-begin,Cpp_prepare_s=times[0],Cpp_commit_s=times[1],candidate_hits=hits-before,numeric=reference_metrics[trial][rank],free_total_bytes=list(torch.cuda.mem_get_info(rank)))
     r['calls'].append(rec)
     if item['phase'].endswith('QA'):
      # A real subsequent transaction consumes committed compressor/cache state.
      forward(item['past']+1,1);r.setdefault('nextstep_commit_checks',[]).append(dict(trial=trial,passed=True))
     barrier.wait(120)
     if rank==0:save()
    r['torch_peak_allocated']=torch.cuda.max_memory_allocated(rank);r['torch_peak_reserved']=torch.cuda.max_memory_reserved(rank)
   except BaseException as e:
    r.update(error=repr(e),traceback=traceback.format_exc());barrier.abort();raise
   finally:
    if ctx:check(f('destroy')(ctx))
    check(f('shutdown')());r['normal_close']=True;save()
  save()
  try:
   with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(owner,range(8)))
   report.update(completed=True,normal=all(o['normal_close']for o in report['owners']))
   for past in [8496,8575]:
    rows=[]
    for i,item in enumerate(plan()):
     if item['past']==past and item['phase']=='measurement':rows.append(dict(trial=i,mode=item['mode'],max8_CAPI_wall_s=max(o['calls'][i]['CAPI_wall_s']for o in report['owners']),max8_Cpp_prepare_commit_s=max(o['calls'][i]['Cpp_prepare_s']+o['calls'][i]['Cpp_commit_s']for o in report['owners']),same_phase_checked_enclosure_s=max(o['calls'][i]['measurement_end_s']for o in report['owners'])-min(o['calls'][i]['measurement_begin_s']for o in report['owners'])))
    report.setdefault('matched_measurements',{})[str(past)]=rows
  except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc());raise
  finally:report['driver_seconds']=time.perf_counter()-began;save()
  print(json.dumps(dict(completed=report['completed'],normal=report['normal'],result=str(out))))

if __name__=='__main__':main()
