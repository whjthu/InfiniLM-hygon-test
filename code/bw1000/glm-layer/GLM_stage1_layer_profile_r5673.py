"""One actual layer38 TP8; synthetic hidden with frozen460 logical history.
NOT EXECUTED. Profile intervals include enqueue idle; no whole-Model TPS claim.
"""
import pathlib,json,ctypes,os,time,hashlib,contextlib,statistics
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');base=r/'candidates/glm-stage1-ledger-r5674';V=ctypes.c_void_p;Z=ctypes.c_size_t
def api():
 lib=ctypes.CDLL(str(base/'libglm_stage1_ledger_r5674.so'))
 lib.glm_layer_id_size_r5670.restype=Z;lib.glm_layer_id_r5670.argtypes=[V]
 lib.glm_layer_create_r5670.argtypes=[ctypes.c_int,ctypes.c_char_p,V];lib.glm_layer_create_r5670.restype=V
 for name in ['error','report']:getattr(lib,'glm_layer_'+name+'_r5670').restype=ctypes.c_char_p
 lib.glm_layer_keys_r5670.argtypes=[V];lib.glm_layer_keys_r5670.restype=ctypes.c_char_p
 lib.glm_layer_load_r5670.argtypes=[V,ctypes.c_char_p]
 for name in ['finish','reset']:getattr(lib,'glm_layer_'+name+'_r5670').argtypes=[V]
 lib.glm_layer_forward_r5670.argtypes=[V,Z,Z,ctypes.POINTER(Z),V,V,V,ctypes.c_int]
 lib.glm_layer_destroy_r5670.argtypes=[V,ctypes.POINTER(ctypes.c_int)]
 lib.glm_layer_control_r5670.argtypes=[V,V]
 return lib

def worker(rank,unique):
 torch.set_num_threads(1);torch.cuda.set_device(rank)
 torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 for name in ['allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']:setattr(torch.backends.cuda.matmul,name,False)
 cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=32;os.sched_setaffinity(0,cpus[4*rank:4*rank+4])
 proof=json.loads((base/'CPU-build.json').read_text());manifest=json.loads((r/'incoming/GLM-stage1-layer-GPU-manifest-r5673.json').read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
 assert proof['completed']and proof['fresh_M_equal_MD']and proof['ABI']==1 and proof['library_SHA256']==manifest['DSO_SHA256'][proof['library']]==sha(proof['library'])
 for path,h in manifest['DSO_SHA256'].items():assert sha(path)==h,path
 lib=api();lib.modern_glm_dense_mode_r48_set.argtypes=[ctypes.c_int]
 class DlInfo(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',V),('sname',ctypes.c_char_p),('saddr',V)]
 process=ctypes.CDLL(None);process.dladdr.argtypes=[V,ctypes.POINTER(DlInfo)];process.dladdr.restype=ctypes.c_int
 def symbol_owner(name,path):
  info=DlInfo();assert process.dladdr(ctypes.cast(getattr(lib,name),V),ctypes.byref(info));actual=pathlib.Path(info.fname.decode()).resolve();assert actual==pathlib.Path(path).resolve(),(name,str(actual),path);return str(actual)
 owners={'layer_create':symbol_owner('glm_layer_create_r5670',proof['library']),'native_grouped_gate':symbol_owner('glm_grouped_gate_r5123','/refactor/candidates/glm-hybrid-decode-r5386/libnative.so')}
 maps={line.split()[-1]for line in pathlib.Path('/proc/self/maps').read_text().splitlines()if line.split()[-1].startswith('/')and'.so'in line.split()[-1]}
 for name,h in proof['SDK_libraries_SHA256'].items():
  matches=[v for v in maps if pathlib.Path(v).name.startswith(name)];assert matches and all(sha(v)==h for v in matches),(name,matches)
 dense=json.loads((r/'results/glm-balanced-micro8-r153-lock.json').read_text())['mode'];assert dense in[0,2];assert lib.modern_glm_dense_mode_r48_set(dense)==0
 cfg=json.loads((model/'config.json').read_text());assert cfg['mlp_layer_types'][38]=='sparse'and cfg['indexer_types'][38]=='full'
 goldpath=r/'incoming/GLM-lowbatch-frozen-gold-r8219.json';gold=json.loads(goldpath.read_text());lengths=[len(a)for a in gold['input_ids']];assert lengths==[91,94,91,85,93,86,82,88,86,95,82,80,96,79,85,99]
 node=os.uname().nodename;p=r/'results'/f'glm-stage1-layer-r5673-{node}-rank{rank}.json';assert not p.exists()
 q={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'rank':rank,'layer':38,'node':node,'cases':[],'input_scope':'Synthetic BF16 hidden, frozen460 logical rows/prompt lengths/positions only; not actual Model hidden or real routing distribution. No wholeModel performance or finaltoken qualification.','gold_SHA256':hashlib.sha256(goldpath.read_bytes()).hexdigest(),'prompt_lengths':lengths,'dense_mode':dense,'environment':{n:os.environ.get(n)for n in ['GLM_GROUPED_ROUTED_R5123','GLM_DECODE_ROW_DEDUP_R5386','GLM_BATCH_ALLSELECTED_R5096','GLM_EXACT_BANKS_R4188','GLM_PROJECTED_HOST_R4504']},'CPU_affinity':sorted(os.sched_getaffinity(0)),'actual_symbol_owners':owners,'actual_mapped_SDK':sorted(maps),'library_SHA256':proof['library_SHA256'],'prefill_diagnostic':[]}
 def save():t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)
 def ck(code):assert code==0,(code,lib.glm_layer_error_r5670().decode())
 def agree(local):
  vote=torch.tensor([float(bool(local))],dtype=torch.float32,device='cuda');torch.cuda.synchronize();ck(lib.glm_layer_control_r5670(ctx,V(vote.data_ptr())));assert vote.item()==8,('allowner_quality_vote',rank,float(vote.item()))
 started=time.perf_counter();uid=ctypes.create_string_buffer(unique,len(unique));ctx=lib.glm_layer_create_r5670(rank,json.dumps(cfg).encode(),uid);assert ctx,lib.glm_layer_error_r5670();q['create_seconds']=time.perf_counter()-started;save()
 keys=json.loads(lib.glm_layer_keys_r5670(ctx));index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];prefix='model.layers.38.';assert len(keys)==len(set(keys))and all(prefix+k in index for k in keys)
 q['state_keys']=len(keys);loadedbytes=0;started=time.perf_counter()
 with contextlib.ExitStack()as stack:
  handles={}
  for first in range(0,len(keys),24):
   held=[];desc=[]
   for key in keys[first:first+24]:
    name=index[prefix+key]
    if name not in handles:handles[name]=stack.enter_context(safe_open(str(model/name),framework='pt',device='cpu'))
    x=handles[name].get_tensor(prefix+key)
    if x.dtype in[torch.float8_e4m3fn,torch.float8_e5m2]:x=x.view(torch.uint8)
    assert x.dtype in[torch.float32,torch.bfloat16,torch.uint8];x=x.contiguous();held.append(x);loadedbytes+=x.numel()*x.element_size()
    desc.append({'name':key,'shape':list(x.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[x.dtype],'pointer':x.data_ptr()})
   ck(lib.glm_layer_load_r5670(ctx,json.dumps(desc).encode()))
 ck(lib.glm_layer_finish_r5670(ctx));agree(torch.cuda.mem_get_info()[0]>=6*1024**3);q.update(load_seconds=time.perf_counter()-started,global_source_bytes=loadedbytes,only_one_layer_weight_load=True,free_after_load=torch.cuda.mem_get_info()[0]);save()
 # Cache all small synthetic inputs once. CPU generator makes TP owners exact
 # replicated hidden without relying on per-device random kernel equivalence.
 cache={}
 def hidden(key,N):
  if key not in cache:
   g=torch.Generator(device='cpu').manual_seed(567300+key);cache[key]=torch.randn((N,6144),generator=g,dtype=torch.float32).to(torch.bfloat16).cuda()
  return cache[key]
 def call(rows,past,counts,seed,profile):
  assert not torch.cuda.is_current_stream_capturing();N=sum(counts);x=hidden(seed,N);pos=torch.tensor([v for a,n in zip(past,counts)for v in range(a,a+n)],dtype=torch.int64,device='cuda');out=torch.empty((N,6144),dtype=torch.bfloat16,device='cuda')
  assert x.is_contiguous()and pos.is_contiguous()and out.is_contiguous();d=(Z*(3*len(rows)))(*[v for triple in zip(rows,past,counts)for v in triple]);torch.cuda.synchronize()
  ck(lib.glm_layer_forward_r5670(ctx,N,len(rows),d,V(x.data_ptr()),V(pos.data_ptr()),V(out.data_ptr()),profile));report=json.loads(lib.glm_layer_report_r5670())
  agree(torch.isfinite(out).all().item()and len(report['route_histogram'])==256 and sum(report['route_histogram'])==N*8 and report['N']==N and report['B']==len(rows)and report['profile']==profile and report['route_readback_outside_wall']and bool(report['stages'])==bool(profile));return out,report
 def restore(extra):
  ck(lib.glm_layer_reset_r5670(ctx))
  for cohort in range(2):
   rows=list(range(cohort*8,cohort*8+8));_,record=call(rows,[0]*8,lengths[cohort*8:cohort*8+8],1000+cohort,0)
   if extra==0 and len(q['prefill_diagnostic'])<2:q['prefill_diagnostic'].append(record)
  for step in range(extra):
   for cohort in range(2):
    rows=list(range(cohort*8,cohort*8+8));call(rows,[lengths[v]+step for v in rows],[1]*8,2000+2*step+cohort,0)
 # Histories and fresh input are restored identically per timed trial; ranks
 # are eight participants, never eight independent temporal repetitions.
 for extra in [0,63,126]:
  samples=[];observers=[];reference={};numerics=[]
  for rep,profile in enumerate([0,0,1,0,1,0]):
   restore(extra)
   for cohort in range(2):
    rows=list(range(cohort*8,cohort*8+8));out,rec=call(rows,[lengths[v]+extra for v in rows],[1]*8,8000+2*extra+cohort,profile)
    if cohort not in reference:reference[cohort]=out.clone()
    else:
     ref=reference[cohort];delta=(out.float()-ref.float()).abs();m=float(delta.max().item());scale=float(ref.float().abs().max().item());agree(m<=0.03125+0.005*scale)
     numerics.append({'rep':rep,'cohort':cohort,'max_abs':m,'reference_max_abs':scale,'rawbits_diagnostic_equal':torch.equal(out.view(torch.uint8),ref.view(torch.uint8))})
    rec.update(rep=rep,cohort=cohort,actual_prefix_lengths=[lengths[v]+extra+1 for v in rows]);(observers if profile else samples).append(rec)
  q['cases'].append({'extra_decode_steps':extra,'profileOFF':samples,'profileON_observer_only':observers,'numeric_comparisons':numerics,'same_initial_history_each_trial':True,'reset_prefill_history_excluded_from_timed_wall':True,'profileOFF_temporal_trials_per_cohort':4,'postcase_free_bytes':torch.cuda.mem_get_info()[0]});save()
 ck(lib.glm_layer_reset_r5670(ctx));expired=ctypes.c_int();ck(lib.glm_layer_destroy_r5670(ctx,ctypes.byref(expired)));assert expired.value==1
 torch.cuda.synchronize();a=torch.ones(32,device='cuda');assert (a+1).sum().item()==64;del a,cache;torch.cuda.synchronize()
 q.update(completed=True,Runtime_weak_expired=True,normal_reset_close=True,Torch_postclose=True,NOT_RUN=['actual Model hidden','CPU route GEMM oracle','wholeModel tokens','PP repartition','BF16 predecode','INT8']);save()

if __name__=='__main__':
 assert torch.cuda.device_count()==8;torch.cuda.set_device(0);lib=api();uid=ctypes.create_string_buffer(lib.glm_layer_id_size_r5670());assert lib.glm_layer_id_r5670(uid)==0
 mp.spawn(worker,args=(bytes(uid),),nprocs=8,join=True)
