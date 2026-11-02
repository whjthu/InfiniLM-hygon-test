"""Two independent same-card pairs; real selected KV weights, synthetic inputs."""
import pathlib,json,hashlib,os,time,signal,ctypes,gc,weakref,traceback,statistics,numpy as np
from tiles import load
from reference import f32,reference,metric,converted_stats
R=pathlib.Path('/refactor');S=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 rank=int(os.environ['SLURM_LOCALID']);assert rank in (0,1);layer=(0,4)[rank];step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/f'Kimi380-gpu-v1-{step}-rank{rank}.json';assert not out.exists()
 remaining=int(float(os.environ['KIMI380_GPUSTART_UNIX'])+570-time.time());assert remaining>30;signal.alarm(remaining)
 q=dict(completed=False,passed=False,actual_step=step,rank=rank,layer=layer,Model_loaded=False,full_draft_loaded=False,target_head_loaded=False,synthetic_inputs=True,quality_or_acceptance_qualified=False,Model_gain='UNKNOWN',cases=[],normal_close=False);started=time.perf_counter();modules={};refs=[];torch=None;arrays={};original=guard=weight=None;old_flags={}
 def save():
  tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\n');tmp.replace(out)
 try:
  mp=S/'GPU_MANIFEST.json';assert sha(mp)==os.environ['KIMI380_GPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items());assert all(sha(p)==h for p,h in m['production_dependencies_SHA256'].items())
  cp=R/'candidates/Kimi-draft-KV-guarded-FP16-r380-cpu-v1/CPU-result.json';assert sha(cp)==m['CPU761_receipt_SHA256'];cpu=json.loads(cp.read_bytes());assert cpu['passed']and cpu['actual_CPU']=='910407.761'and not cpu['GPU_initialized']and not cpu['Model_loaded']
  cm=R/'incoming/Kimi-draft-KV-guarded-FP16-r380/cpu-v1/SOURCE_MANIFEST.json';assert sha(cm)==cpu['source_manifest_SHA256']==m['CPU761_source_manifest_SHA256'];cm=json.loads(cm.read_bytes());assert all(sha(S/n)==cm['files_SHA256'][n]for n in ['guarded_linear.py','tiles.py','reference.py']);assert all(sha(p)==h for p,h in cm['dependencies_SHA256'].items())
  import torch as t
  torch=t
  from guarded_linear import GuardedFP16Linear
  torch.set_num_threads(1);torch.set_num_interop_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0);assert 'gfx936'in props.gcnArchName
  assert sha(torch._C.__file__)==cpu['Torch_C_SHA256'];q.update(Torch_version=torch.__version__,Torch_C_SHA256=sha(torch._C.__file__),HIP_version=torch.version.hip,device=dict(visible_device=0,physical_mask=os.environ['HIP_VISIBLE_DEVICES'],name=props.name,arch=props.gcnArchName),memory_before=list(torch.cuda.mem_get_info()))
  assert os.environ['HIP_VISIBLE_DEVICES']==str(rank)and q['memory_before'][0]>(12<<30)
  for name in ['allow_fp16_reduced_precision_reduction','allow_fp16_accumulation']:
   try:old_flags[name]=getattr(torch.backends.cuda.matmul,name);setattr(torch.backends.cuda.matmul,name,False)
   except AttributeError:pass
  q['matmul_policy']=dict(original_FP16_flags=old_flags,effective_FP16_flags={n:getattr(torch.backends.cuda.matmul,n)for n in old_flags},BF16_reduced_precision_unchanged=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,actual_ISA_accumulation_precision='UNKNOWN; numeric tested, not inferred from API flags/scalar FP16 specs')
  assert all(v is False for v in q['matmul_policy']['effective_FP16_flags'].values())
  arrays,proof=load();assert proof['tensors']==cpu['checkpoint']['tensors'];assert proof['file_bytes']==cpu['checkpoint']['file_bytes']and proof['file_mtime_ns']==cpu['checkpoint']['file_mtime_ns'];q['checkpoint']=proof
  for projection,suffix in [('KVA','kv_a_proj_with_mqa.weight'),('KVB','kv_b_proj.weight')]:
   key=f'layers.{layer}.self_attn.{suffix}';bits=arrays[key];torch.cuda.synchronize();tick=time.perf_counter();weight=torch.from_numpy(bits.copy()).view(torch.bfloat16).to('cuda:0');original=torch.nn.Linear(bits.shape[1],bits.shape[0],bias=False,device='meta',dtype=torch.bfloat16);original.weight=torch.nn.Parameter(weight,requires_grad=False);torch.cuda.synchronize();load_wall=time.perf_counter()-tick
   tick=time.perf_counter();guard=GuardedFP16Linear(original);torch.cuda.synchronize();copy_wall=time.perf_counter()-tick;assert guard.weight16 is not None
   refs.extend(weakref.ref(v)for v in [guard,original,original.weight,guard.weight16]);modules[projection]=guard
   halfbits=guard.weight16.cpu().view(torch.uint16).numpy().copy();expected=f32(bits).astype(np.float16).view(np.uint16);stored=halfbits.view(np.float16).astype(np.float32);assert np.isfinite(stored).all()
   q.setdefault('weight_loads',[]).append(dict(key=key,BF16_H2D_and_module_wall_s=load_wall,once_FP16_weight_guard_copy_drain_wall_s=copy_wall,private_FP16_bytes=guard.weight16.numel()*2,CPU_to_GPU_FP16_storage_exact_diagnostic=bool(np.array_equal(halfbits,expected)),GPU_storage_different_from_CPU_half_count=int(np.count_nonzero(halfbits!=expected)),GPU_storage_nonzero_to_zero=int(np.count_nonzero((bits&0x7fff!=0)&(stored==0))),FP16_conversion=converted_stats(f32(bits))))
   del halfbits,expected
  original=guard=weight=None;q['prepare_wall_s']=time.perf_counter()-started;save()
  def timed(g,x,mode):
   torch.cuda.synchronize();before=dict(g.counts);tick=time.perf_counter();y=g(x,enabled=True)if mode else g.original(x);torch.cuda.synchronize();wall=time.perf_counter()-tick
   if not mode:g.counts['baseline']+=1;g.last_path='baseline'
   path=g.last_path;delta={k:g.counts[k]-before[k]for k in before};del y
   return dict(mode=mode,wall_s=wall,path=path,count_delta=delta)
  def numeric(g,x,bits,mode):
   torch.cuda.synchronize();y=g(x,enabled=True)if mode else g.original(x);torch.cuda.synchronize()
   if not mode:g.counts['baseline']+=1;g.last_path='baseline'
   finite=bool(torch.isfinite(y).all());q['last_qualification']=dict(mode=mode,path=g.last_path,whole_output_finite=finite);save();assert g.last_path==('hit'if mode else'baseline')and finite
   rows=np.unique(np.array([0,x.shape[0]//2,x.shape[0]-1]));cols=np.unique(np.linspace(0,y.shape[1]-1,32,dtype=int));xb=x[rows.tolist()].cpu().view(torch.uint16).numpy().copy();a=y[rows.tolist()][:,cols.tolist()].cpu().float().numpy().copy();ref=reference(xb,bits[cols].copy(),bool(mode));v=metric(a,ref);qual=dict(independent_CPU=v,rows=rows.tolist(),columns=cols.tolist(),reference='NumPyF32 fullK plus explicit FP16 storage and BF16 rounding',path=g.last_path);q['last_qualification'].update(qual);save();assert v['finite']and v['NRMSE']<.02
   return y,qual
  def bench_case(projection,M,index):
   case_start=time.perf_counter();g=modules[projection];key=f'layers.{layer}.self_attn.'+('kv_a_proj_with_mqa.weight'if projection=='KVA'else'kv_b_proj.weight');bits=arrays[key];N,K=bits.shape;seed=380000+rank*100+index;torch.manual_seed(seed);torch.cuda.reset_peak_memory_stats();x=torch.randn((M,K),dtype=torch.bfloat16,device='cuda:0')*(.1 if projection=='KVA'else 1.);input_before=hashlib.sha256(x.cpu().view(torch.uint16).numpy().tobytes()).hexdigest();assert np.array_equal(g.original.weight.cpu().view(torch.uint16).numpy(),bits)
   q['active_case']=dict(projection=projection,M=M,K=int(K),N=int(N),key=key,seed=seed);save()
   # Actual BF16 baseline must pass independent CPU before the first candidate.
   ref,base=numeric(g,x,bits,0);on,candidate=numeric(g,x,bits,1)
   # Whole output comparison uses GPU F32 only; scalar metric is not Modelquality.
   rf=ref.float();of=on.float();delta=of-rf;nr=float(torch.linalg.vector_norm(delta).item()/(torch.linalg.vector_norm(rf).item()+1e-30));maxabs=float(delta.abs().amax().item());q['last_whole_output_compare']=dict(NRMSE=nr,maxabs=maxabs);save();assert np.isfinite(nr)and nr<.02;del ref,on,rf,of,delta
   row=dict(projection=projection,key=key,M=int(M),K=int(K),N=int(N),seed=seed,input_scale=.1 if projection=='KVA'else 1.,input_kind='synthetic; M32646 current uncached KV geometry, M8 small suffix geometry not measured fullquery',input_SHA256=input_before,baseline_qualification=base,candidate_qualification=candidate,whole_output_GPU_F32_NR=nr,whole_output_maxabs=maxabs,warm=[],groups=[],candidate_all_hits=True)
   q['active_case']=row
   for mode in [0,1,0,1]:row['warm'].append(timed(g,x,mode))
   for group in range(5):
    arms=[timed(g,x,mode)for mode in [0,1,1,0]];off=(arms[0]['wall_s']+arms[3]['wall_s'])/2;onwall=(arms[1]['wall_s']+arms[2]['wall_s'])/2;drift=abs(arms[3]['wall_s']-arms[0]['wall_s'])/off;row['groups'].append(dict(group=group,arms=arms,OFF_s=off,ON_s=onwall,ratio=onwall/off,OFF_endpoint_drift=drift));save()
   row['candidate_all_hits']=all(v['path']==('hit'if v['mode']else'baseline')for v in row['warm']for _ in [0])and all(a['path']==('hit'if a['mode']else'baseline')for z in row['groups']for a in z['arms']);assert row['candidate_all_hits']
   assert hashlib.sha256(x.cpu().view(torch.uint16).numpy().tobytes()).hexdigest()==input_before and np.array_equal(g.original.weight.cpu().view(torch.uint16).numpy(),bits);row.update(input_and_weight_state_unchanged=True,median_OFF_s=statistics.median(v['OFF_s']for v in row['groups']),median_ON_s=statistics.median(v['ON_s']for v in row['groups']),max_OFF_endpoint_drift=max(v['OFF_endpoint_drift']for v in row['groups']),measurement_call_count=20,measured_wall_sum_s=sum(a['wall_s']for z in row['groups']for a in z['arms']),whole_case_including_fixture_qual_state_and_checks_wall_s=time.perf_counter()-case_start,memory_checkpoint_after=list(torch.cuda.mem_get_info()),memory_checkpoint_not_peak=True,Torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),Torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_scope='Torch allocator since this case input generation, includes numerical comparison/F32 metric buffers and both arms; not physical memory or candidate-only peak');del x;return row
  with torch.inference_mode():
   for index,(projection,M)in enumerate([('KVA',32646),('KVB',32646),('KVA',8),('KVB',8)]):q['cases'].append(bench_case(projection,M,index));q.pop('active_case',None);save()
   # One real-weight unsafe-input probe per projection, outside all ABBA intervals.
   q['range_fallback_probes']=[]
   for projection in ['KVA','KVB']:
    g=modules[projection];z=torch.full((1,g.original.in_features),65536.,device='cuda:0',dtype=torch.bfloat16);before=dict(g.counts);torch.cuda.synchronize();tick=time.perf_counter();y=g(z,enabled=True);torch.cuda.synchronize();wall=time.perf_counter()-tick;assert g.last_path=='range_fallback'and bool(torch.isfinite(y).all());q['range_fallback_probes'].append(dict(projection=projection,path=g.last_path,whole_guard_fallback_drain_wall_s=wall,count_delta={k:g.counts[k]-before[k]for k in before}));del z,y
  q['final_counts']={k:dict(g.counts)for k,g in modules.items()};q['passed']=len(q['cases'])==4
  g=None
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  original=guard=weight=g=z=y=None;modules.clear();arrays.clear();gc.collect()
  cleanup_errors=[]
  if torch is not None:
   try:
    for n,v in old_flags.items():setattr(torch.backends.cuda.matmul,n,v)
    if torch.cuda.is_initialized():torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.synchronize();q['memory_after_cleanup_checkpoint']=list(torch.cuda.mem_get_info())
   except BaseException as e:cleanup_errors.append(repr(e))
  q['cleanup_errors']=cleanup_errors;q['all_owned_module_parameter_weakrefs_expired']=all(v()is None for v in refs);q['normal_close']=not cleanup_errors and q['all_owned_module_parameter_weakrefs_expired']
  q.update(completed=True,whole_driver_wall_s=time.perf_counter()-started,source_manifest_SHA256=os.environ['KIMI380_GPU_MANIFEST_SHA'],timer_scope='Perarm synchronous complete wrapper wall. Candidate includes percall finite/maxguard,inputcast,F.linear,outputfinite,backcast,drain. Oneprivateweightconversion+H2D/CPUoracle/load separately. No GPU F64, no fullModel/head, no scalarFP16spec speed prediction.');save()
 assert q['passed']and q['normal_close'],q.get('error')
if __name__=='__main__':main()
