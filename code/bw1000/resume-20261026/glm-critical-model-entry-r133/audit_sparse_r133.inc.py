def audit_sparse(q,stage,rank):
 arms=q['episodes'];expected=[3,3,1,1];profile=arms[1]['sparse_observer'];assert profile['enabled']and profile['events_allocated']==512 and profile['required']==expected
 for a in [arms[0],arms[2]]:assert not a['sparse_observer']['enabled']and a['sparse_observer']['samples']==[]
 got=[0,0,0,0];calls=arms[1]['native_calls'];seen=set();layer_begin=[0,38][stage];layer_end=[38,78][stage]
 for sample in profile['samples']:
  seq=sample['sequence'];assert type(seq)is int and 1<=seq<=len(calls)and seq not in seen;seen.add(seq);call=calls[seq-1];event=q['native_events'][call['event_index']];triples=[list(x)for x in zip(call['rows'],call['pasts'],call['counts'])];assert sample['stage']==stage and sample['rank']==rank and sample['row_past_count']==triples and sample['tokens']==event['input_tokens'];assert event['pasts']==call['pasts']and event['counts']==call['counts']
  if call['op']=='DECODE'and sample['tokens']in [16,15]:kind=0 if sample['tokens']==16 else 1;assert all(p>0 and n==1 for _,p,n in triples)
  elif call['op']=='PREFIX'and len(triples)in [8,1]:kind=2 if len(triples)==8 else 3;assert all(p==0 and n>0 for _,p,n in triples)
  else:raise AssertionError(('r131 mismatched natural selector',sample,call))
  got[kind]+=1;marks=sample['marks'];assert len(marks)==sample['event_count']<=512 and len(marks)>=2;assert marks[0]['name']=='CAPI.owner.begin'and marks[-1]['name']=='CAPI.owner.end.before_original_drain';assert all(math.isfinite(x['owner_host_s'])and math.isfinite(x['gpu_from_first_s'])and x['gpu_from_first_s']>=0 for x in marks);assert all(b['owner_host_s']>=a['owner_host_s']and b['gpu_from_first_s']>=a['gpu_from_first_s']-1e-7 for a,b in zip(marks,marks[1:]));names=[x['name']for x in marks]
  for l in range(layer_begin,layer_end):assert sum(n.startswith('layer.'+str(l)+'.')and n.endswith('.begin')for n in names)==1 and names.count('layer.'+str(l)+'.end')==1
  assert all(names.count(n)==1 for n in ['PP.incoming.begin','PP.incoming.end','PP.outgoing.begin','PP.outgoing.end']);full=sample['full_detail_layer'];shared=sample['shared_detail_layer'];assert layer_begin<=full<layer_end and layer_begin<=shared<layer_end and full!=shared and 'layer.'+str(full)+'.full.begin'in names and 'layer.'+str(shared)+'.shared.begin'in names
  detail=[n for n in names if n.startswith('detail.layer.')];assert detail and {int(n.split('.')[2])for n in detail}<={full,shared}
  for l in [full,shared]:
   for phase in ['RMS.attention','attention','RMS.ffn','FFN']:assert 'detail.layer.'+str(l)+'.'+phase+'.begin'in names and 'detail.layer.'+str(l)+'.'+phase+'.end'in names
 assert got==profile['coverage']and all(g<=e for g,e in zip(got,expected));assert profile['sequence']==len(calls)
 return {'stage':stage,'rank':rank,'coverage':got,'coverage_complete':got==expected,'missing_status':'COMPLETE'if got==expected else'UNKNOWN','event_limit':512,'selector_sequence_full_triples_verified':True,'nested_host_event_spans':True}

def analyze_envelopes(episodes):
 # Diagnostic guard only. Useful-token normalization does not remove natural context confounding.
 rows=[]
 for a in episodes:
  s=a['serving'];scores=a['task_scores'];m=a['metrics'];visible=m['useful_visible_output_tokens'];decode=sum(e['native_seconds']for e in _emitter_events(a)if not e['prefill'])
  rows.append({'label':a['label'],'task_tok_s':m['overall_output_tok_s'],'decode_CAPI_tok_s':max(0,visible-32)/decode if decode>0 else None,'visible_tokens':visible,'elapsed_s':s['elapsed_seconds'],'correct':sum(x['flexible_correct']for x in scores),'parse_failed':sum(x['parse_failed']for x in scores),'EOS':sum(x['stop_reason']=='EOS'for x in s['requests']),'caps':sum(x['stop_reason']=='max_new_tokens'for x in s['requests']),'unfinished':s['unfinished']})
 drift={};effect={};passed=True
 for k in ['task_tok_s','decode_CAPI_tok_s']:
  b,p,a=[x[k]for x in rows];mean=(b+a)/2 if b is not None and a is not None else None
  drift[k]=abs(a-b)/mean if mean and b>0 and a>0 else None;effect[k]=abs(p/mean-1)if mean and p is not None else None;passed=passed and drift[k]is not None and effect[k]is not None and drift[k]<=.03 and effect[k]<=.03
 quality=all(x['EOS']==32 and x['caps']==x['unfinished']==0 for x in rows)and rows[1]['correct']>=min(rows[0]['correct'],rows[2]['correct'])and rows[1]['parse_failed']<=max(rows[0]['parse_failed'],rows[2]['parse_failed'])
 return {'episodes':rows,'control_relative_drift':drift,'profile_absolute_relative_effect':effect,'prespecified_tolerance':.03,'perturbation_passed':bool(passed),'quality_passed':bool(quality),'classification':'DIAGNOSTIC_ONLY_NO_SPEEDUP_CLAIM','limitations':'Only one before/profile/after repetition; natural tokens/context may differ; >3percent or uncovered selector stops actionable attribution; GPU event timelines include enqueue/dependency idle; host/stage spans overlap; no F64 or peak BW claims','next_patch_gate':'At most one largest actionable about10percent actual criticalwall interval, beyond measured observer/control drift; no layer multiplication or component to Model extrapolation'}

def _emitter_events(a):
 # Filled from the exact stage1/rank0 report during audit, preserving all original events.
 return a['audit_native_events']
