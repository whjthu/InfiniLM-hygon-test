CONFIG_SHA256='9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213'
INDEX_SHA256='a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd'
PARTITION_SHA256='7cf1c0fa877680656cca6e875be1f0107fe3c1455ff631e9622dce758380a7eb'
import collections,datetime,hashlib,json,math,pathlib,re,struct
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';model=pathlib.Path.home()/'shared/Kimi-K3';cfg=json.loads((model/'config.json').read_text())['text_config'];mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];headers={};metadata={};header_bytes=0
for name in sorted(set(v for k,v in mapping.items()if k.startswith('language_model.'))):
 with(model/name).open('rb')as f:
  n=struct.unpack('<Q',f.read(8))[0];assert 0<n<150000000;raw=f.read(n);assert len(raw)==n;header_bytes+=n;assert header_bytes<1500000000;h=json.loads(raw)
 headers[name]=hashlib.sha256(raw).hexdigest()
 for k in h:
  if k in mapping and k.startswith('language_model.'):metadata[k]=h[k]
assert all(k in metadata for k in mapping if k.startswith('language_model.'))
sizes={'F32':4,'BF16':2,'F16':2,'U8':1,'I8':1,'I64':8,'I32':4,'F64':8,'BOOL':1};ranges={pp:[[max(0,min(93,95*p//pp-1)),max(0,min(93,95*(p+1)//pp-1))]for p in range(pp)]for pp in [2,4]};reports=[]
for tp,pp in [(16,2),(16,4)]:
 stages=[]
 for stage,(begin,end)in enumerate(ranges[pp]):
  buckets=collections.Counter();counts=collections.Counter();unclassified=collections.Counter()
  for key,v in metadata.items():
   layer=re.search(r'language_model\.model\.layers\.(\d+)\.',key)
   if layer:
    if not begin<=int(layer[1])<end:continue
   elif key=='language_model.model.embed_tokens.weight':
    if stage!=0:continue
   elif key.startswith(('language_model.lm_head.','language_model.model.norm.','language_model.model.output_attn_res_')):
    if stage!=pp-1:continue
   else:continue
   b=math.prod(v['shape'])*sizes[v['dtype']];shard=False;category='replicated_or_conservative'
   if '.block_sparse_moe.experts.'in key:shard=True;category='MXFP4_experts'
   elif '.block_sparse_moe.shared_experts.'in key or'.mlp.'in key:shard=True;category='shared_or_dense_MLP'
   elif '.self_attn.'in key:
    component=key.split('.self_attn.',1)[1].split('.',1)[0]
    if component in ['q_proj','k_proj','v_proj','q_b_proj','kv_b_proj','g_proj','o_proj','f_b_proj','b_proj','q_conv1d','k_conv1d','v_conv1d','A_log','dt_bias']:shard=True;category='sharded_attention'
    elif component not in ['f_a_proj','q_a_proj','q_a_layernorm','kv_a_proj_with_mqa','kv_a_layernorm','o_norm']:unclassified[component]+=b
   if key.endswith('.self_attn.A_log'):b=cfg['linear_attn_config']['num_heads']*4
   if shard:assert b%tp==0,(key,b,tp);b//=tp
   # Runtime uses F32 convolution and output norm even for lower precision checkpoint inputs.
   if any('.self_attn.'+name+'.'in key for name in ['q_conv1d','k_conv1d','v_conv1d','o_norm'])and sizes[v['dtype']]<4:b=b*4//sizes[v['dtype']]
   if key.endswith('.self_attn.A_log'):b=cfg['linear_attn_config']['num_heads']//tp*4
   buckets[category]+=b;counts[category]+=1
  total=sum(buckets.values());stages.append({'PP_stage':stage,'layers':[begin,end],'planning_reserve_bytes':6*1024**3,'parameter_plus_reserve_fits':total+6*1024**3<64*1024**3,'parameter_bytes_conservative':total,'parameter_GiB_conservative':total/1024**3,'category_bytes':dict(buckets),'category_keys':dict(counts),'unclassified_attention_replicated':dict(unclassified),'remaining_of_64GiB_before_cache_workspace_allocator':64-total/1024**3})
 reports.append({'TP':tp,'PP':pp,'nodes':tp//8*pp,'cards':tp*pp,'stages':stages,'all_stages_parameter_plus_6GiB_reserve_fits_64GiB':all(v['parameter_GiB_conservative']+6<64 for v in stages)})
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'method':'Read checkpoint headers only; actual source Kimi Experts /TP, shared/denseMLP /TP, explicit attention columns/rows /TP; replicated router/down/up/norm/f_a/q_a/kv_a and full embedding. Conservatively full LMhead tail, unclassified attention replicated. Conv/o_norm F32, A_log trimmed actual heads. Not exact allocator peak/cache/workspace/posttranspose memory; sixGiB only planning reserve, actual all32 GPU mem_get_info/load/finish/pre-fill still required.','metadata_keys':len(metadata),'header_bytes_read':header_bytes,'header_sha256':headers,'source_commit':'b7b37077b3a014a3a1c35b5ef07ca91722dbbef0','topologies':reports,'model_forward_or_serving_qualified':False}
d.update(served_layers=93,config_SHA256=hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),index_SHA256=hashlib.sha256((model/'model.safetensors.index.json').read_bytes()).hexdigest(),partition_header_SHA256=PARTITION_SHA256,PP_ranges_by_size={str(k):v for k,v in ranges.items()},header_only_not_GPU_capacity=True,model_host_view=str(model),source_commit='5f91ad738583db8f5617aeae53b76523f85d1ec6');assert d['config_SHA256']==CONFIG_SHA256 and d['index_SHA256']==INDEX_SHA256; p=r/'results/Kimi-93layer-static-budget-r9152.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps(d))
