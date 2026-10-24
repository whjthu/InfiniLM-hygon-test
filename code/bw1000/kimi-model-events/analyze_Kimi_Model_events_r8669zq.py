"""CPU-only consumer of finite all-owner whole-Model stream phase records.
No active-kernel inference, foreign-clock subtraction, or sum-of-rank maxima.
"""
import math,statistics

def analyze(reports):
 assert len(reports)==4;rows=[]
 for node,q in enumerate(reports):
  raw=q['whole_model_owner_profile'];assert raw['all8_published']and len(raw['owners'])==8
  for device,v in sorted(raw['owners'].items(),key=lambda p:int(p[0])):
   device=int(device);stage=node//2;lo,hi=[(0,46),(46,93)][stage]
   assert(v['device'],v['global_TP_rank'],v['TP'],v['PP_stage'],v['PP'])==(device,(node%2)*8+device,16,stage,2)
   assert v['read_after_original_reset_drain']and 0<v['events_used']<=640 and len(v['calls'])==3
   for c in v['calls']:
    pts=c['points'];assert pts[0]['name']=='task.begin'and pts[-1]['name']=='task.return'
    inner=[p for p in pts if p['layer']>=0]
    expected=[(layer,name)for layer in range(lo,hi)for name in ['decoder.entry','attention.end','FFN.begin','FFN.end']]
    assert[(p['layer'],p['name'])for p in inner]==expected
    assert c['forward_ordinal']in[0,40,41];phases={}
    for p in pts[1:]:
     for k in ['stream_span_ms','host_span_ms','threadCPU_span_ms']:assert math.isfinite(p[k])and p[k]>=0
     a,b=p['from'],p['name']
     if a=='task.begin':key='CPU_metadata_before_Model'
     elif a=='model.begin':key='embedding_or_PP_receive'
     elif a=='decoder.entry':key='attention_input_residual_norm_and_attention'
     elif a=='attention.end':key='post_attention_residual_norm'
     elif a=='FFN.begin':key='routed_MoE_full'if p['is_MoE']else'dense_FFN_full'
     elif b=='decoder.entry':key='layer_output_residual'
     elif b=='model.return':key='stage_output_norm_head_or_PP_send'
     elif b=='task.return':key='sampling_IDcopy_original_drain_or_non_tail_return'
     else:raise AssertionError((a,b))
     z=phases.setdefault(key,{'stream_span_ms':0.,'host_span_ms':0.,'threadCPU_span_ms':0.})
     for k in z:z[k]+=p[k]
    assert abs(sum(z['stream_span_ms']for z in phases.values())-c['whole_task_stream_span_ms'])<.05
    rows.append({'node':node,'device':device,'global_TP_rank':v['global_TP_rank'],'stage':stage,'forward_ordinal':c['forward_ordinal'],'host_task_return_ms':c['host_task_return_ms'],'threadCPU_task_return_ms':c['threadCPU_task_return_ms'],'original_tail_drain_ms':c['original_tail_drain_ms'],'whole_task_stream_span_ms':c['whole_task_stream_span_ms'],'phases':phases})
 assert len(rows)==96
 summaries=[]
 for call in [0,40,41]:
  for stage in range(2):
   group=[z for z in rows if z['forward_ordinal']==call and z['stage']==stage];assert len(group)==16
   slow_stream=max(group,key=lambda z:z['whole_task_stream_span_ms']);slow_host=max(group,key=lambda z:z['host_task_return_ms'])
   summaries.append({'forward_ordinal':call,'stage':stage,'stream_span_ms_min_median_max':[min(v['whole_task_stream_span_ms']for v in group),statistics.median(v['whole_task_stream_span_ms']for v in group),max(v['whole_task_stream_span_ms']for v in group)],'same_owner_slowest_stream_vector':slow_stream,'same_owner_slowest_host_return_vector':slow_host})
 return {'all32owners_3calls_schema_pass':True,'per_stage_call_summaries':summaries,'owner_vectors':rows,'limits':['Stream includes enqueue/queue/dependency/CCL wait, not activekernel.','Non-tailPPtaskreturn is host enqueue completion, not GPUcomplete; no newper-call sync.','Outer PP/embedding/head categories intentionally combined; no TP-vsPPpurebandwidth claim.','Do not sumdifferentownermax orcrosshosttimestamps. Host/threadCPU/stream overlap; never add.','No perlayer routinghist/providerW13-W2 split captured by these2hostTUs.']}
