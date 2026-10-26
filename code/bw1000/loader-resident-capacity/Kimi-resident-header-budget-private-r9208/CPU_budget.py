import pathlib,json,hashlib,struct,math,re,collections,datetime,os
r=pathlib.Path('/refactor');model=pathlib.Path('/shared/Kimi-K3');cf=(model/'config.json').read_bytes();ix=(model/'model.safetensors.index.json').read_bytes();assert hashlib.sha256(cf).hexdigest()=='9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213';assert hashlib.sha256(ix).hexdigest()=='a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd'
mp=json.loads(ix)['weight_map'];stats=collections.defaultdict(collections.Counter);counts=collections.Counter();headerbytes=0;headers={};ignored=collections.Counter()
for filename in sorted(set(v for k,v in mp.items()if k.startswith('language_model.'))):
 with(model/filename).open('rb')as f:
  n=struct.unpack('<Q',f.read(8))[0];assert 0<n<150000000;raw=f.read(n);assert len(raw)==n;headerbytes+=n;assert headerbytes<1500000000;h=json.loads(raw)
 headers[filename]=hashlib.sha256(raw).hexdigest()
 for k,v in h.items():
  if k not in mp or not k.startswith('language_model.'):continue
  assert mp[k]==filename
  layer=re.search(r'language_model\.model\.layers\.(\d+)\.',k)
  if layer:
   lid=int(layer[1])
   if not 0<=lid<93:ignored['nonserved_layer_tensors']+=1;continue
   group='layer:'+str(lid)
  elif k=='language_model.model.embed_tokens.weight':group='embed'
  elif k.startswith(('language_model.lm_head.','language_model.model.norm.','language_model.model.output_attn_res_')):group='tail'
  else:ignored['unmapped_nonlayer_tensors']+=1;continue
  b=v['data_offsets'][1]-v['data_offsets'][0];assert b>=0;stats[group][v['dtype']]+=b;counts[group]+=1
out=[];capacity=128000*1024**2;reserve=8*1024**3
for pp,ranges in [(2,[(0,46),(46,93)]),(4,[(0,24),(24,47),(47,71),(71,93)])]:
 for stage,(lo,hi)in enumerate(ranges):
  keys=['layer:'+str(i)for i in range(lo,hi)]+(['embed']if stage==0 else[])+(['tail']if stage==pp-1 else[]);tot=collections.Counter()
  for key in keys:assert counts[key]>0,key;tot.update(stats[key])
  nbytes=sum(tot.values());out.append({'TP':16,'PP':pp,'stage':stage,'layers':[lo,hi],'unsharded_stage_checkpoint_payload_bytes':nbytes,'unsharded_stage_checkpoint_payload_GiB':nbytes/1024**3,'source_dtype_bytes':dict(tot),'selected_keys':sum(counts[k]for k in keys),'current_node_Slurm_memory_bytes':capacity,'planning_nonresident_reserve_bytes':reserve,'full_stage_memfd_plus_reserve_fits_current_node_request':nbytes+reserve<capacity,'nodes_per_TP_stage':2,'no_per_rank_slice_lower_bound_claim':True})
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU_submitted':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'header_only_no_payload_read':True,'header_bytes_read':headerbytes,'headers_SHA256':headers,'config_SHA256':hashlib.sha256(cf).hexdigest(),'index_SHA256':hashlib.sha256(ix).hexdigest(),'served_layers':93,'stages':out,'ignored_tensor_counts':dict(ignored),'scope':'Exact stored tensor byte ranges grouped by explicitly served mainlayers/embed/tail; not runtime perrank allocator bytes. Existing loader each node opens fullstage tensors before native TP split. InfiniLoader canonicalstage memfd capacity necessary check only; no RSS/PSS/cgroup/device utilization or speedup claim. Explicit physicalnode/rank sharded adapter remains work; currentruntime expects global tensor shapes.'}
(r/'results/Kimi-resident-header-budget-r9208.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
