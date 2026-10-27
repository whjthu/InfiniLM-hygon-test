"""Audit actual all16 normal outputs and same-load arrival ABBA, no exact-token gate."""
import pathlib,json,hashlib,math
R=pathlib.Path.home()/'hygon-inference-refactor-20261014';D=R/'incoming/GLM-arrival-defer-r148';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def audit():
 launch=json.loads((R/'GLM-arrival-defer-r148-launch.json').read_bytes());run=launch['run'];m=json.loads((D/'GPU-manifest.json').read_bytes());assert sha(D/'GPU-manifest.json')==launch['manifest_SHA256'];cases=json.loads((D/'corpus-reference.json').read_bytes())['requests'];shared=None;owners=[];emitter=None
 for stage in [0,1]:
  for rank in range(8):
   p=R/'results'/f'glm-arrival-r148-{run}-stage{stage}-rank{rank}.json';q=json.loads(p.read_bytes());assert all(q[k]for k in ['completed','worker_shutdown_completed','post_close_Torch_liveness','normal_reset_close','benchmark_complete','warm_CCL_progress_passed']);assert q['library_sha256']==m['Model_SHA256']and q['cache_after_destroy']['active']==q['cache_after_destroy']['active_bytes']==0 and q['minimum_driver_free_bytes']>=512*1024**2;assert q['host_bank_after_destroy_r4504'][:3]==[38 if stage==0 else 40]*2+[0]
   assert not q['observer_before_destroy']['enabled']and q['observer_native_destroy_returned0'];arms=q['episodes'];assert [(a['label'],a['deferred_feedback'])for a in arms]==[('A1-serial',False),('B1-deferred',True),('B2-deferred',True),('A2-serial',False)];hashes=[a['command_SHA256']for a in q['warmups']+arms]
   if shared is None:shared=hashes
   assert shared==hashes,'all16 command identity'
   for a in q['warmups']+arms:
    assert a['complete']and a['vector_counters']['actual_verified']and a['vector_counters']['delta']==a['vector_counters']['expected'];assert not a['sparse_observer']['enabled']and a['sparse_observer']['samples']==[];assert not a['m15_oproj'];cmds=a['command_trace'];assert cmds[-1]['op']=='DONE'and[cmd['seq']for cmd in cmds]==list(range(1,len(cmds)+1));ledger=[0]*32;anchors=[0]*32;corpus=[dict(c,input_ids=c['input_ids'][:40],max_new_tokens=2)for c in cases]if a['warm']else cases
    byseq={c['seq']:c for c in cmds};seen={};calls=a['native_calls']
    for call in calls:
     cmd=byseq[call['seq']];items=[x for x in cmd['items']if x[0]in call['rows']];assert call['op']==cmd['op']and call['rows']==[x[0]for x in items]and call['pasts']==[x[1]for x in items]and call['counts']==[x[2]for x in items];assert len(items)<=(16 if cmd['op']=='DECODE'else 8);seen.setdefault(cmd['seq'],[]).extend(x[0]for x in items)
     e=q['native_events'][call['event_index']];assert e['rows']==call['rows']and e['input_tokens']==sum(call['counts'])and e['prefill']==(cmd['op']!='DECODE')and math.isfinite(e['native_seconds'])and e['native_seconds']>=0 and e['ragged_counter_delta']==e['expected_ragged_counter_delta']
     for row,past,n,_ in items:
      assert ledger[row]==past and n>=1 and past+n<=640;L=len(corpus[row]['input_ids']);assert a['warm']or cmd['emitter_time_ms']/1000>=4*(row//8)-.001
      if cmd['op']=='PREFIX':assert past+n<=L-1
      elif cmd['op']=='ANCHOR':assert past==L-1 and n==1 and anchors[row]==0;anchors[row]+=1
      else:assert n==1 and past>=L and anchors[row]==1
      ledger[row]+=n
    for cmd in cmds:
     if cmd['op']in ['PREFIX','ANCHOR','DECODE']:assert seen[cmd['seq']]==[x[0]for x in cmd['items']]
    assert anchors==[1]*32
    if stage==1 and rank==0:
     serving=a['serving'];assert serving['all_requests_completed']and serving['global_reset_allowed']and serving['unfinished']==serving['not_arrived']==0 and serving['peak_admitted_live']<=32 and not serving['row_recycling']
     for i,(req,rec)in enumerate(zip(corpus,serving['requests'])):assert rec['id']==req['id']and rec['arrival']==(0 if a['warm']else 4*(i//8))and rec['service_start']>=rec['admitted']>=rec['arrival']and ledger[i]==len(req['input_ids'])+len(rec['generated_ids'])-1
   owners.append({'stage':stage,'rank':rank,'SHA256':sha(p),'actual_step':q['actual_producer'],'normal':True})
   if stage==1 and rank==0:emitter=q
 rows=[]
 for a in emitter['episodes']:
  decode=sum(emitter['native_events'][c['event_index']]['native_seconds']for c in a['native_calls']if c['op']=='DECODE');metrics=a['metrics'];scores=a['task_scores'];s=a['serving'];visible=metrics['useful_visible_output_tokens'];rows.append({'label':a['label'],'deferred':a['deferred_feedback'],'task_tok_s':metrics['overall_output_tok_s'],'decode_CAPI_tok_s':max(0,visible-32)/decode,'visible_tokens':visible,'task_seconds':s['elapsed_seconds'],'correct':sum(x['flexible_correct']for x in scores),'EOS':sum(r['stop_reason']=='EOS'for r in s['requests']),'caps':sum(r['stop_reason']=='max_new_tokens'for r in s['requests']),'metrics':metrics,'peak_live':s['peak_admitted_live']})
 return {'qualified':True,'execution_qualified':True,'performance_gain_qualification':'NOT_AUTOMATIC_REVIEW_ABBA_QUALITY_AND_REAL_TWO_MICRO_COVERAGE','owners':owners,'episodes':rows,'paired_quality':emitter['paired_quality'],'scope':'Same32 arrival0/4/8/12 maxlive32 M16x2 currentce830 ABBA; naturalEOS notHTTP; no model gain from component ratio','token_identity_gate':False,'Model_SHA256':m['Model_SHA256']}
if __name__=='__main__':print(json.dumps(audit()))
