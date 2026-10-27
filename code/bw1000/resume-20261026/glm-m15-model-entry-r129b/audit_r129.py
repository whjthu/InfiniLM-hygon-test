import pathlib,json,hashlib,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';inc=r/'incoming/GLM-M15-Model-r129';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def audit():
 launch=json.loads((r/'GLM-M15-Model-r129-launch.json').read_bytes());m=json.loads((inc/'GPU-manifest.json').read_bytes());assert sha(inc/'GPU-manifest.json')==launch['manifest_SHA256'];run=launch['run'];result={'qualified':False,'run':run,'owners':[],'Model_SHA256':m['Model_SHA256'],'token_identity_gate':False,'scope':'finite32 syntheticburst 0/4/8/12 nativearrivalemulation, NOTHTTP, a549parent unchangedmath exceptqualifiedrow2down'}
 for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in m['existing_remote_SHA256'].items():assert sha(r/n)==h,n
 cases=json.loads((inc/'corpus-reference.json').read_bytes())['requests'];arrivals=[4*(i//8)for i in range(32)];shared=None
 for stage in [0,1]:
  for rank in range(8):
   path=r/'results'/f'glm-arrival-r129-{run}-stage{stage}-rank{rank}.json';q=json.loads(path.read_bytes());assert all(q[k]for k in ['completed','worker_shutdown_completed','post_close_Torch_liveness','normal_reset_close','benchmark_complete','warm_CCL_progress_passed']);assert q['library_sha256']==m['Model_SHA256']and q['token_exactness_gate']is False;assert q['cache_after_destroy']['active']==q['cache_after_destroy']['active_bytes']==0;assert q['minimum_driver_free_bytes']>=512*1024**2;assert q['host_bank_after_destroy_r4504'][:3]==[38 if stage==0 else 40]*2+[0]
   released=q['M15_after_destroy'];assert released[0]==0 and released[7]==released[8]and released[12]==released[13]==0 and released[5]==released[16]==0
   assert q['M15_default_OFF_verified']and q['M15_invalid_wrongthread_untouched'];assert q['M15_T16_same_load_controls']and all(x['default_nonselection_verified']and all(x['M15_delta17'][i]==0 for i in [1,2,3,5,15,16])for x in q['M15_T16_same_load_controls'])
   assert len(q['warmups'])==2 and all(z['complete']for z in q['warmups']);arms=q['episodes'];assert [(e['label'],e['chunked'],e['down_row2'],e['m15_oproj'])for e in arms]==[('A0-M15OFF',False,True,False),('B0-M15ON',False,True,True),('B1-M15ON',False,True,True),('A1-M15OFF',False,True,False)]
   hashes=[e['command_SHA256']for e in q['warmups']+arms]
   if shared is None:shared=hashes
   assert hashes==shared,'all16 WORLDcommandtrace identical'
   for e in q['native_events']:assert e['ragged_counter_delta']==e['expected_ragged_counter_delta']and math.isfinite(e['native_seconds'])and e['native_seconds']>=0 and e['input_tokens']<=4096
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
   result['owners'].append({'stage':stage,'rank':rank,'SHA256':sha(path),'actual_step':q['actual_producer'],'minimum_driver_free_bytes':q['minimum_driver_free_bytes'],'normal':True,'all_episode_commands':hashes})
   if stage==1 and rank==0:result.update(episodes=arms,paired_quality=q['paired_quality'],warmups=q['warmups'])
 result['qualified']=True;return result
if __name__=='__main__':print(json.dumps(audit()))
