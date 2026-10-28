"""Same-owner adjacent stream spans only. Never infer pure busy/PP/CCL."""
def summarize(raw):
 assert raw['all8_published'];owners=[]
 for device,o in raw['owners'].items():
  for c in o['calls']:
   by={};rows=[]
   for p in c['points']:
    if 'from'not in p:continue
    pair=(p['from'],p['name']);name='other_including_dependencies'
    if pair==('decoder.entry','attention.end'):name='KDA_with_AttnRes_norm_projections_TPwait'if p['is_KDA']else'MLA_with_AttnRes_norm_projections_TPwait'
    if pair==('FFN.begin','FFN.end'):name='MoE_routed_shared_TPwait'if p['is_MoE']else'dense_FFN_TPwait'
    if p['is_MoE'] and p.get('from_layer')==p['layer'] and (p['from']=='FFN.begin' or p['from'].startswith('MoE.')) and (p['name']=='FFN.end' or p['name'].startswith('MoE.')):name='MoE_routed_shared_TPwait'
    v=float(p['stream_span_ms']);assert v>=0;by[name]=by.get(name,0.)+v;rows.append({'from':p['from'],'to':p['name'],'layer':p['layer'],'stream_span_ms':v,'scope':name})
   owners.append({'device':int(device),'PP_stage':o['PP_stage'],'TP_rank':o['global_TP_rank'],'forward_ordinal':c['forward_ordinal'],'whole_task_stream_ms':c['whole_task_stream_span_ms'],'original_tail_drain_ms':c['original_tail_drain_ms'],'adjacent_disjoint_stream_ms_by_scope':by,'adjacent_intervals':rows})
 critical={}
 for o in owners:
  stage=str(o['PP_stage'])
  if stage not in critical or o['whole_task_stream_ms']>critical[stage]['whole_task_stream_ms']:critical[stage]=o
 return {'local_owner_calls':owners,'critical_owner_within_each_present_PPstage':critical,'crosshost_clock_subtraction':False,'pure_GPU_busy_CCL_PP_head':'UNKNOWN','warning':'event spans include enqueue/dependency/communication waits. FFN includes shared/routed work; detailed operator split unavailable. Do not add parent/child spans.'}
