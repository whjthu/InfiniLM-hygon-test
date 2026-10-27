import pathlib,json,hashlib,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';inc=r/'incoming/GLM-critical-Model-r133';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def audit():
 launch=json.loads((r/'GLM-critical-Model-r133-launch.json').read_bytes());m=json.loads((inc/'GPU-manifest.json').read_bytes());assert sha(inc/'GPU-manifest.json')==launch['manifest_SHA256'];run=launch['run'];result={'qualified':False,'run':run,'owners':[],'Model_SHA256':m['Model_SHA256'],'token_identity_gate':False,'scope':'finite32 syntheticburst 0/4/8/12 nativearrivalemulation, NOTHTTP, a549parent unchangedmath exceptqualifiedrow2down'}
 for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in m['existing_remote_SHA256'].items():assert sha(r/n)==h,n
 cases=json.loads((inc/'corpus-reference.json').read_bytes())['requests'];arrivals=[4*(i//8)for i in range(32)];shared=None
 for stage in [0,1]:
  for rank in range(8):
   path=r/'results'/f'glm-arrival-r133-{run}-stage{stage}-rank{rank}.json';q=json.loads(path.read_bytes());assert all(q[k]for k in ['completed','worker_shutdown_completed','post_close_Torch_liveness','normal_reset_close','benchmark_complete','warm_CCL_progress_passed']);assert q['library_sha256']==m['Model_SHA256']and q['token_exactness_gate']is False;assert q['cache_after_destroy']['active']==q['cache_after_destroy']['active_bytes']==0;assert q['minimum_driver_free_bytes']>=512*1024**2;assert q['host_bank_after_destroy_r4504'][:3]==[38 if stage==0 else 40]*2+[0]
   released=q['M15_after_destroy'];assert released[0]==0 and released[7]==released[8]and released[12]==released[13]==0 and released[5]==released[16]==0
   assert q['M15_default_OFF_verified']and q['M15_invalid_wrongthread_untouched'];assert q['M15_T16_same_load_controls']and all(x['default_nonselection_verified']and all(x['M15_delta17'][i]==0 for i in [1,2,3,5,15,16])for x in q['M15_T16_same_load_controls'])
   assert q['observer_profile_saved_before_OFF']and q['observer_native_destroy_returned0']and not q['observer_before_destroy']['enabled'];assert len(q['warmups'])==1 and all(z['complete']for z in q['warmups']);arms=q['episodes'];assert [(e['label'],e['chunked'],e['down_row2'],e['m15_oproj'])for e in arms]==[('controls-before',False,True,False),('sparse-profile',False,True,False),('controls-after',False,True,False)]
   hashes=[e['command_SHA256']for e in q['warmups']+arms]
   if shared is None:shared=hashes
   assert hashes==shared,'all16 WORLDcommandtrace identical'
   for e in q['native_events']:assert len(e['existing_pipeline_profile'])==6 and all(math.isfinite(x)and x>=0 for x in e['existing_pipeline_profile']);assert e['ragged_counter_delta']==e['expected_ragged_counter_delta']and math.isfinite(e['native_seconds'])and e['native_seconds']>=0 and e['input_tokens']<=4096
   for a in q['warmups']+arms:
    md=a['m15_provider'];expected=sum(q['native_events'][c['event_index']]['input_tokens']==15 for c in a['native_calls'])*[38,40][stage];dd=md['delta'];on=a['m15_oproj'];assert md['actual_owner_verified']and md['expected_eligible_projection_calls']==expected and dd[1]==expected and dd[2]==(expected if on else 0)and dd[3]==(0 if on else expected)and dd[5]==dd[16]==0 and dd[15]==dd[2]
    vc=a['vector_counters'];assert vc['actual_verified']and vc['delta']==vc['expected']and vc['routed_layer_count']==[35,40][stage];assert a['complete']and a['native_ordinary_checks_kept'];cmds=a['command_trace'];calls=a['native_calls'];assert [c['seq']for c in cmds]==list(range(1,len(cmds)+1));assert cmds[-1]['op']=='DONE';forwards=[c for c in cmds if c['op']in ['PREFIX','ANCHOR','DECODE']];assert len(forwards)==len(calls)==a['native_event_end']-a['native_event_start'];ledger=[0]*32;anchors=[0]*32;decodes=[0]*32;corpus=[dict(c,input_ids=c['input_ids'][:40],max_new_tokens=2)for c in cases]if a['warm']else cases
    for c,call in zip(forwards,calls):
     items=c['items'];rows=[x[0]for x in items];pasts=[x[1]for x in items];counts=[x[2]for x in items];assert (c['seq'],c['op'],rows,pasts,counts)==(call['seq'],call['op'],call['rows'],call['pasts'],call['counts']);assert len(rows)==len(set(rows))and len(rows)<=(16 if c['op']=='DECODE'else 8)
     event=q['native_events'][call['event_index']];assert event['rows']==rows and event['input_tokens']==sum(counts)and event['prefill']==(c['op']!='DECODE')
     for row,past,n,_ in items:
      assert ledger[row]==past and n>=1 and past+n<=640;L=len(corpus[row]['input_ids']);assert a['warm']or c['emitter_time_ms']/1000>=arrivals[row]-.001
      if c['op']=='PREFIX':assert past+n<=L-1 and(not a['chunked']or n<=32)
      elif c['op']=='ANCHOR':assert n==1 and past==L-1 and anchors[row]==0;anchors[row]+=1
      else:assert n==1 and past>=L and anchors[row]==1;decodes[row]+=1
      ledger[row]+=n
    assert anchors==[1]*32
    if stage==1 and rank==0:
     s=a['serving'];assert s['all_requests_completed']and s['global_reset_allowed']and s['unfinished']==s['not_arrived']==0 and s['peak_admitted_live']<=16 and not s['row_recycling'];assert len(s['requests'])==32
     for i,(req,rec)in enumerate(zip(corpus,s['requests'])):
      assert rec['id']==req['id']and rec['row']==i and rec['status']=='completed'and rec['stop_reason']in ['EOS','max_new_tokens'];assert rec['arrival']==(0. if a['warm']else arrivals[i]);assert rec['service_start']>=rec['admitted']>=rec['arrival'];assert 1<=len(rec['generated_ids'])<=req['max_new_tokens'];assert ledger[i]==len(req['input_ids'])+len(rec['generated_ids'])-1 and decodes[i]==len(rec['generated_ids'])-1;stamps=[v['time']for v in rec['events']];assert stamps==sorted(stamps)and all(rec['arrival']<=t<=s['elapsed_seconds']for t in stamps)
     if not a['warm']:assert len(a['task_scores'])==32
   sparse=audit_sparse(q,stage,rank);result.setdefault('sparse_owners',[]).append(sparse)
   result['owners'].append({'stage':stage,'rank':rank,'SHA256':sha(path),'actual_step':q['actual_producer'],'minimum_driver_free_bytes':q['minimum_driver_free_bytes'],'normal':True,'all_episode_commands':hashes})
   if stage==1 and rank==0:
    for a in arms:a['audit_native_events']=[q['native_events'][c['event_index']]for c in a['native_calls']]
    result.update(episodes=arms,paired_quality=q['paired_quality'],warmups=q['warmups'])
 result.update(execution_qualified=True);result['observer_analysis']=analyze_envelopes(result['episodes']);result['coverage_complete']=all(z['coverage_complete']for z in result['sparse_owners']);result['perturbation_passed']=result['observer_analysis']['perturbation_passed'];result['quality_passed']=result['observer_analysis']['quality_passed'];result['qualified']=bool(result['coverage_complete']and result['perturbation_passed']and result['quality_passed']);return result
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

if __name__=='__main__':print(json.dumps(audit()))

