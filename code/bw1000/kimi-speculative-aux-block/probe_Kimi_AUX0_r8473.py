import os,pathlib,json
assert os.environ.get('KIMI_AUX0_ROOT_GO_R8473')=='1'
assert os.environ.get('KIMI_AUX0_WATCH_OWNER_R8473')
with pathlib.Path('/refactor/results/Kimi-speculative-AUX0-r8473.run.lock').open('x') as f: json.dump({'watch_owner':os.environ['KIMI_AUX0_WATCH_OWNER_R8473'],'pid':os.getpid(),'step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','')},f)
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['INFINILM_KIMI_SPEC_MATH_R3676']='11'
import pathlib,json,ctypes,time,collections,hashlib,datetime,gc,statistics
import numpy as np
import torch
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/shared/Kimi-K3');b=r/'candidates/kimi-K7-MoE-CPU-r8104';torch.set_num_threads(1);node=os.uname().nodename
layers=json.loads(os.environ['KIMI_MOE_LAYERS_R4512']);assert layers and all(isinstance(x,int)and 1<=x<93 for x in layers)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();build=json.loads((r/'results/Kimi-K7-MoE-CPU-build-r8104.json').read_text());assert build['completed_CPU_compile_link']and not build['dlopen_performed'];assert sha(b/'libmodern_Kimi_K7_MoE_r8104.so')==build['library_SHA256']=='282b1d028c07a373b52176f36e99da75ba663d8aa236e98459a282b65d335f29'
for name,z in build['direct_link_fingerprints_before_and_after'].items():assert sha(pathlib.Path(name))==z['SHA256'],name
for rows in build['two_TU_complete_MD_dependencies'].values():
 for z in rows:assert sha(pathlib.Path(z['path']))==z['SHA256'],z['path']
assert layers==[20]
fixture_names=['verify-call396']
mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];config=json.loads((model/'config.json').read_text());assert config['text_config']['num_experts']==896 and config['text_config']['num_experts_per_token']==16
lib=ctypes.CDLL(str(b/'libmodern_Kimi_K7_MoE_r8104.so'));lib.modern_kimi_moe_replay_r8082.argtypes=[ctypes.c_char_p];lib.modern_kimi_moe_replay_result_r8082.restype=ctypes.c_char_p
class DlInfo(ctypes.Structure):
 _fields_=[('filename',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('symbol_address',ctypes.c_void_p)]
dl=ctypes.CDLL(None);dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];dl.dladdr.restype=ctypes.c_int;bindings={}
for name in ['modern_kimi_moe_replay_r8082','modern_kimi_moe_replay_result_r8082','modern_kimi_router_counts_r4635','modern_kimi_moe_aux_counts_r4430','modern_kimi_route_chunk_counts_r4435','modern_kimi_stable_down_counts_r4012']:
 z=DlInfo();assert dl.dladdr(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(z))and z.filename;path=pathlib.Path(z.filename.decode()).resolve();assert path==(b/'libmodern_Kimi_K7_MoE_r8104.so').resolve();bindings[name]={'path':str(path),'SHA256':sha(path)}
flags={'INFINILM_KIMI_SPEC_MATH_R3676':'11','INFINILM_KIMI_ROUTER_M1_R4635':'1','INFINILM_KIMI_MOE_AUX_M1_R4430':'1','INFINILM_DENSE_PREFILL_F32_R2475':'1'}
dl.getenv.argtypes=[ctypes.c_char_p];dl.getenv.restype=ctypes.c_char_p
assert all(os.environ.get(k)==v and dl.getenv(k.encode())==v.encode()for k,v in flags.items())
mapped={p:sha(pathlib.Path(p))for p in{v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(n in v for n in ['libinfiniops.so','libinfinirt.so','libinfiniccl.so','libinfinilm_runtime.so','libmodern_kimi_dense0_r4346.so','libmodern_kimi_joint_expert_r4620.so','libkimi_grouped_prefill.so'])}}
for name,z in build['direct_link_fingerprints_before_and_after'].items():
 if any(n in name for n in ['libinfiniops.so','libinfinirt.so','libinfiniccl.so','libinfinilm_runtime.so','libmodern_kimi_dense0_r4346.so','libmodern_kimi_joint_expert_r4620.so','libkimi_grouped_prefill.so']):assert [h for p,h in mapped.items()if pathlib.Path(p).resolve()==pathlib.Path(name).resolve()]==[z['SHA256']],name
for layer,call in [(L,C)for L in layers for C in [396]]:
 out=r/'results'/f'kimi-moe-AUX0-r8473-{node}-L{layer}-C{call}.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':node,'layer':layer,'completed':False,'phase':'CPU_single_layer_load_actual_flags','producer_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'library_sha256':build['library_SHA256'],'scope':'C396 real captured N8 L20 TP8 component; only captured TP0/6 inputs verified/equal; remaining six owners receive same input. AUX1/0/0/1; no full Model or serving qualification.'}
 def save():
  p=out.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(out)
 save();stage=0 if layer==20 else 3;traceroot=r/'results'/f'trained-features-r4809-trainedModal4809-f71640dc8e2f73073d20-call{call}';fixtures={};proofs=[]
 for rank in [0,6]:
  f=traceroot/f'stage{stage}-rank{rank}-layer{layer}-ffn_norm-full-bf16.bin';z=np.fromfile(f,np.uint16).reshape(-1,7168);assert z.shape==(8,7168);fixtures[rank]={'ffn_norm':z};proofs.append({'file':str(f),'shape':list(z.shape),'sha256':sha(f)})
 assert np.array_equal(fixtures[0]['ffn_norm'],fixtures[6]['ffn_norm'])
 prefix=f'language_model.model.layers.{layer}.block_sparse_moe.';groups=collections.defaultdict(list)
 for key in mapping:
  if key.startswith(prefix):groups[mapping[key]].append(key)
 # Metadata-only budget gate in the actual task/container before checkpoint tensors.
 def memory_guard():
  result={'required_weight_bytes':16102205952,'reserve_bytes':8*1024**3,'ancestors':[],'passed':False}
  try:
   cg=pathlib.Path('/proc/self/cgroup').read_text();result['proc_self_cgroup']=cg
   paths=[v.split(':',2)[2]for v in cg.splitlines()if v.startswith('0::')];assert len(paths)==1,'cgroupv2 unresolved'
   root=pathlib.Path('/sys/fs/cgroup').resolve();current=(root/paths[0].lstrip('/')).resolve();assert current.is_relative_to(root)
   headrooms=[]
   while True:
    row={'path':str(current)}
    for name in ['memory.current','memory.max']:
     z=current/name;row[name]=z.read_text().strip()if z.is_file()else'NOT_AVAILABLE'
    result['ancestors'].append(row)
    if row['memory.max'].isdigit():
     assert row['memory.current'].isdigit();headrooms.append(int(row['memory.max'])-int(row['memory.current']))
    else:assert row['memory.max']=='max' or current==root,'ancestor memory.max unavailable'
    if current==root:break
    current=current.parent
   mem=dict((line.split(':',1)[0],line.split(':',1)[1].strip())for line in pathlib.Path('/proc/meminfo').read_text().splitlines());available=int(mem['MemAvailable'].split()[0])*1024
   result['MemAvailable_bytes']=available;result['minimum_ancestor_headroom_bytes']=min(headrooms)if headrooms else None
   result['admitted_headroom_bytes']=min([available]+headrooms);result['required_total_bytes']=result['required_weight_bytes']+result['reserve_bytes'];result['passed']=result['admitted_headroom_bytes']>=result['required_total_bytes']
  except Exception as exc:result['error']=repr(exc)
  return result
 d['CPU_memory_guard']=memory_guard();save()
 if not d['CPU_memory_guard']['passed']:
  d.update(phase='NOT_RUN_CPU_MEMORY_BUDGET',completed=False,normal_NOT_RUN=True,no_retry=True);save();raise SystemExit(0)
 held=[];descriptors=[];tick=time.perf_counter();nbytes=0
 for filename,keys in sorted(groups.items()):
  with safe_open(model/filename,framework='pt',device='cpu')as f:
   for key in sorted(keys):
    w=f.get_tensor(key).clone().contiguous();assert w.dtype in[torch.float32,torch.bfloat16,torch.uint8];held.append(w);descriptors.append({'name':key[len(prefix):],'pointer':w.data_ptr(),'shape':list(w.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[w.dtype]});nbytes+=w.numel()*w.element_size()
 d['actual_symbol_bindings']=bindings;d['actual_library_mappings']=mapped;d['actual_flags']=flags;d['CPU_build_receipt_SHA256']=sha(r/'results/Kimi-K7-MoE-CPU-build-r8104.json');d['fixture_labels']=fixture_names;d['component_policy']='actual4635 math11 routerM1/auxM1 flags, original MXFP4, chunk8 CCL; fullFN bits diagnostic only';d['scope']='Actual4809 C396 N8 L20 checkpoint TP8, group0 only, AUX1/0/0/1. No Model finaltoken or serving claim.'
 d.update(CPU_weight_seconds=time.perf_counter()-tick,CPU_weight_bytes=nbytes,CPU_parameters=len(held),fixture_proof=proofs,phase='resident_native_profile');save();inputs=[];trials=[];x=fixtures[0]['ffn_norm']
 def add(label,M,layout,spec,serial=False,diagnostic=False,mask=0,group=0):
  z=np.ascontiguousarray(np.tile(x,(int(np.ceil(M/8)),1))[:M]if layout=='actual8_tiled'else np.repeat(x[:1],M,axis=0));inputs.append(z);trials.append({'label':label,'tokens':M,'input_pointer':z.ctypes.data,'mask':mask,'group':group,'serial':serial,'spec':spec,'diagnostic':diagnostic,'iterations':5})
 # Optimization OFF/ON/ON/OFF is legacy AUX flag 1/0/0/1.
 # One CPU checkpoint read; four synchronous native single-layer load/close cycles.
 d['arms']=[];outputs={};routes={};d['native_single_layer_load_count']=4;d['full_target_Model_load_count']=0
 def counters():
  result={}
  for name,width in [('modern_kimi_router_counts_r4635',3),('modern_kimi_stable_down_counts_r4012',3),('modern_kimi_route_chunk_counts_r4435',3),('modern_kimi_moe_aux_counts_r4430',6)]:
   fn=getattr(lib,name);fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];result[name]=[]
   for rank in range(8):
    a=(ctypes.c_uint64*width)();assert fn(rank,a,width)==0;result[name].append(list(a))
  return result
 for arm,aux in [('baseline-before',1),('candidate-first',0),('candidate-repeat',0),('baseline-after',1)]:
  os.environ['INFINILM_KIMI_MOE_AUX_M1_R4430']=str(aux);assert dl.getenv(b'INFINILM_KIMI_MOE_AUX_M1_R4430')==str(aux).encode()
  inputs.clear();trials.clear();add(arm,8,'actual8_tiled',True,diagnostic=True,group=0)
  rawdir=r/'results'/f'kimi-moe-AUX0-r8473-{node}-L{layer}-C{call}-{arm}-raw';rawdir.mkdir(exist_ok=False)
  payload={'config':config,'layer':layer,'weights':descriptors,'raw_directory':str(rawdir),'trials':trials}
  before=counters();tick=time.perf_counter();status=lib.modern_kimi_moe_replay_r8082(json.dumps(payload).encode());q=json.loads(lib.modern_kimi_moe_replay_result_r8082());after=counters()
  entry={'arm':arm,'aux_flag':aux,'optimization_on':aux==0,'native_invocation_seconds':time.perf_counter()-tick,'producer_report':q,'raw_directory':str(rawdir)};d['arms'].append(entry);save()
  assert status==0 and q['completed'] and q['normal_all8_Runtime_owners_closed_before_communicators'] and q['communication_group_destroyed'] and len(q['reports'])==8
  delta={name:[[b-a for a,b in zip(x,y)]for x,y in zip(before[name],after[name])]for name in before};entry['counter_delta']=delta
  assert delta['modern_kimi_router_counts_r4635']==[[30,240,30]]*8
  assert delta['modern_kimi_stable_down_counts_r4012']==[[30,240,30]]*8
  assert delta['modern_kimi_route_chunk_counts_r4435']==([[30,240,30]]*8 if aux else [[0,0,0]]*8)
  assert delta['modern_kimi_moe_aux_counts_r4430']==([[30,240,30]*2]*8 if aux else [[0]*6]*8)
  outputs[arm]={};routes[arm]={};entry['tracefree_per_owner_ms']=[]
  for rr in q['reports']:
   rank=rr['rank'];assert rr['native_Runtime_owner_shutdown_completed'] and rr['ModelWeights_destroyed_before_Runtime'] and len(rr['cases'])==1
   cs=rr['cases'][0];assert cs['group']==0 and cs['grouped_call_delta']==0 and cs['tokens']==8 and cs['last_output_finite'];points=collections.defaultdict(list)
   for desc in cs['raw']:
    p=rawdir/desc['file'];assert p.stat().st_size==desc['bytes'];desc['sha256']=sha(p);a=np.fromfile(p,{'BF16':np.uint16,'F32':np.float32,'I32':np.int32}[desc['dtype']]).reshape(desc['shape']);points[desc['point']].append(a.reshape(-1,a.shape[-1]))
   got={k:np.concatenate(v)for k,v in points.items()};v=got['ffn_output'];assert v.shape==(8,7168);f32=(v.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f32).all()
   ri=got['routing_ids'];rw=got['routing_weights'];assert ri.shape==(8,16) and rw.shape==(8,16) and ((ri>=0)&(ri<896)).all() and np.isfinite(rw).all()
   outputs[arm][rank]=v.copy();routes[arm][rank]=(ri.copy(),rw.copy());entry['tracefree_per_owner_ms'].append({'rank':rank,'samples':cs['trace_disabled_host_ms_per_forward'],'median_ms':statistics.median(cs['trace_disabled_host_ms_per_forward']),'complete_FFN_SHA256':hashlib.sha256(v.tobytes()).hexdigest()})
  save()
 d['comparisons']=[]
 for arm in ['candidate-first','candidate-repeat','baseline-after']:
  for rank in range(8):
   a=outputs['baseline-before'][rank];v=outputs[arm][rank];f=(a.astype(np.uint32)<<16).view(np.float32);g=(v.astype(np.uint32)<<16).view(np.float32)
   assert all(np.array_equal(x,y)for x,y in zip(routes['baseline-before'][rank],routes[arm][rank]))
   d['comparisons'].append({'arm':arm,'rank':rank,'full_FFN_bit_exact':bool(np.array_equal(a,v)),'differing_elements':int(np.count_nonzero(a!=v)),'total_elements':int(a.size),'max_abs_difference':float(np.max(np.abs(f-g))),'mean_abs_difference':float(np.mean(np.abs(f-g))),'route_ids_order_and_scores_exact':True})
 d['repeatability_diagnostic']=[{'rank':rank,'baseline_bit_exact':bool(np.array_equal(outputs['baseline-before'][rank],outputs['baseline-after'][rank])),'candidate_bit_exact':bool(np.array_equal(outputs['candidate-first'][rank],outputs['candidate-repeat'][rank]))}for rank in range(8)]
 d['tracefree_ABBA']={}
 for mode,names in [('baseline',['baseline-before','baseline-after']),('candidate',['candidate-first','candidate-repeat'])]:
  rows=[[z['median_ms']for z in arm['tracefree_per_owner_ms']]for arm in d['arms']if arm['arm']in names];paired=[statistics.mean(x)for x in zip(*rows)];d['tracefree_ABBA'][mode]={'perowner_paired_ms':paired,'maxowner_ms':max(paired)}
 d['tracefree_ABBA']['maxowner_ratio']=d['tracefree_ABBA']['baseline']['maxowner_ms']/d['tracefree_ABBA']['candidate']['maxowner_ms']
 d.update(completed=True,phase='completed',scope='C396 N8 L20 checkpoint TP8 AUX1/0/0/1 component only; four native layer load/close cycles. Final Model IDs/EOS, state rollback and user decode speed NOT_TESTED. Complete diagnostic FFN outputs retained; differences quantified without bit-difference auto-rejection.',Model_performance_adopted=False)
 save();print('KIMI_AUX0_R8473_COMPLETED',json.dumps(d['tracefree_ABBA']),flush=True);del held,descriptors,payload,inputs,q;gc.collect()
