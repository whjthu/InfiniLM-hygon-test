"""Independent post-terminal audit; output metadata only, retain raw on server."""
import pathlib,json,hashlib,math,statistics,subprocess,sys,traceback,datetime
R=pathlib.Path.home()/'hygon-inference-refactor-20261014';LABEL='GLM-W4-cross-workload-r349';S=R/'incoming/GLM-W4-cross-workload-r349/runtime_v4/owner';C=R/'incoming/GLM-W4-cross-workload-r349/controller_v3'
ACTUAL='910376.828';KEEPER='910407.682';COMMIT='40830260b66c639d853665d2611f9c4d2438b117'
OWNER_SHA='4e5bbea816c8569fd7fbf382f2efc44787f8937dcf096fbacb889b1c8b917366';CONTROLLER_SHA='4942c2cc401ababe588f070a334d7b05bdf6c7b4a166629a99a8d26effeb9ce2';NATIVE='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'completed':False,'passed':False,'actual_GPU':ACTUAL,'actual_CPU_keeper':KEEPER,'publication_commit':COMMIT,'owner_manifest_SHA256':OWNER_SHA,'controller_manifest_SHA256':CONTROLLER_SHA,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Three complete recorded HTTP bodies,6 calls each; tool-intent generation; buffered HTTP; scoped Model validation only','full_benchmark_or_online_SLO_qualified':False,'client_incremental_token_latency_qualified':False,'fresh_closed_loop_tool_execution_qualified':False,'raw_bodies_replies_history_and_full_IDs_stay_private':True,'cross_node_clock_subtraction':False,'overlapping_stage_sums_used':False,'kernel_busy_or_child_event_attribution':'UNKNOWN'}
def load(path):return json.loads(pathlib.Path(path).read_bytes())
def verify_manifest(directory,pin):
 assert sha(directory/'SOURCE_MANIFEST.json')==pin
 m=load(directory/'SOURCE_MANIFEST.json');assert all(sha(directory/n)==h for n,h in m['files_SHA256'].items())
def means(cases,owner_rows):
 selected=[2,3,4,5];values={}
 for metric in ['HTTP_wall_s','Native_request_wall_s','Native_prefill_CAPI_host_s','Native_decode_CAPI_host_s','completion_tokens']:
  x=[]
  for i in selected:
   r=owner_rows[i];c=cases[i]
   value=c['HTTP_wall_s']if metric=='HTTP_wall_s'else r['wall_s']if metric=='Native_request_wall_s'else r['actual_generated_tokens']if metric=='completion_tokens'else sum(z['Native_CAPI_s']for z in r['native_calls']if z['prefill']==(metric=='Native_prefill_CAPI_host_s'))
   assert math.isfinite(value)and value>0;x.append(value)
  w2=[x[0],x[3]];w4=[x[1],x[2]];a=statistics.mean(w2);b=statistics.mean(w4)
  values[metric]={'units':'tokens'if metric=='completion_tokens'else'seconds','ratio_interpretation':'Output-length diagnostic, not speedup'if metric=='completion_tokens'else'Wall speedup','W2_mean':a,'W4_mean':b,'W2_over_W4':a/b,'W4_reduction_fraction':1-b/a,'adjacent_pair_ratios':[x[0]/x[1],x[3]/x[2]],'both_adjacent_pairs_same_direction':(x[0]>x[1])==(x[3]>x[2]),'W2_endpoint_drift_fraction':(w2[1]-w2[0])/a,'W4_endpoint_drift_fraction':(w4[1]-w4[0])/b,'W2_values':w2,'W4_values':w4}
 return values
try:
 verify_manifest(S,OWNER_SHA);verify_manifest(C,CONTROLLER_SHA);card=load(S/'FIXTURE_CARD.json');fixtures=card['fixtures']
 watchpath=R/'results'/(LABEL+'-watch.json');w=load(watchpath)
 assert w['completed']and w['scoped_Model_pair_qualified']and w['actual_GPU']==ACTUAL and w['CPU_keeper_actual']==KEEPER and w['publication_commit']==COMMIT
 assert w['terminal'][0]==ACTUAL and w['terminal'][2:4]==['COMPLETED','0:0']and w['srun_exit_code']==0 and w['post_empty']
 assert len(w['fresh_before'])==len(w['post'])==4
 for rows in [w['fresh_before'],w['post']]:assert {x['node']for x in rows}=={'a02r4n12','a02r4n13','a02r4n14','a02r4n15'}and all(x['KFD_empty']and x['port43843_bindable']for x in rows)
 s=subprocess.check_output(['sacct','-S','2026-10-24','-j',ACTUAL+','+KEEPER,'-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed'],text=True,timeout=25)
 account={x.split('|')[0]:x.split('|')for x in s.splitlines()if x.split('|')[0]in[ACTUAL,KEEPER]}
 assert set(account)=={ACTUAL,KEEPER}and all(x[2:4]==['COMPLETED','0:0']for x in account.values())
 owners=[];signature=None;tail=None;minimum=[]
 for stage in range(4):
  for rank in range(8):
   path=R/'results'/('GLM-latest-full-owner-r290-'+LABEL+'-stage%d-rank%d.json'%(stage,rank));a=load(path)
   assert all(a[k]for k in ['completed','worker_shutdown_completed','post_close_Torch_liveness','normal_reset_close','native_service_loop_completed','normal_shutdown','observer_native_destroy_returned0'])
   assert a['actual_producer']==ACTUAL and a['actual_step']==ACTUAL and a['source_manifest_SHA256']==OWNER_SHA and a['latest_protocol_binding_r290']['source_manifest_SHA256']==OWNER_SHA and a['library_sha256']==a['Native_SHA256']==NATIVE
   assert a['prepared_entry_identity']['upstream_commit']=='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'and a['prepared_entry_identity']['server_SHA256']=='ef831cc23a8a71ed8d89824187b2494e41c16ec23aee1bd2f8a7794d68fa96de'
   assert a['planned_HTTP']==a['planned_Native']==a['attempted_Native']==a['completed_Native']==a['full_requests_completed']==len(a['service_requests'])==18
   assert a['cache_after_destroy']['created']==a['cache_after_destroy']['destroyed']and a['cache_after_destroy']['active']==a['cache_after_destroy']['active_bytes']==0
   assert a['host_bank_after_destroy_r4504'][:3]==[0,0,0]and a['minimum_driver_free_bytes']>=512*1024**2;minimum.append(a['minimum_driver_free_bytes'])
   sig=[]
   for i,r in enumerate(a['service_requests']):
    f=fixtures[i//6];ai=i%6;window=[2,4,2,4,4,2][ai]
    assert(r['global_index'],r['pair_index'],r['arm_index'],r['fixture_index'],r['fixture_id'])==(i,i,ai,i//6,f['id'])
    assert r['prefill_control_window']==window and r['paired_warmup']==(ai<2)and r['prompt_tokens']==f['current_prepared_prompt_tokens']and r['requested_output_budget']==4096
    assert r['actual_prompt_token_ids_SHA256']==f['current_prepared_prompt_IDs_SHA256']and r['original_reset_all32_completed']and r['generated_natural_EOS']and r['reason']=='eos_token'
    calls=r['native_calls'];assert len(calls)==math.ceil(r['prompt_tokens']/512)+r['generated_tokens']-1
    assert sum(z['T']for z in calls if z['prefill'])==r['prompt_tokens']and sum(not z['prefill']for z in calls)==r['generated_tokens']-1
    assert all(math.isfinite(z['Native_CAPI_s'])and z['Native_CAPI_s']>=0 for z in calls)
    sig.append((r['actual_prompt_token_ids_SHA256'],r['generated_token_ids_SHA256'],r['generated_tokens'],r['reason']))
   if signature is None:signature=sig
   assert sig==signature
   owners.append({'stage':stage,'rank':rank,'path':str(path),'SHA256':sha(path),'Native_requests':18,'normal_destroy_and_shutdown':True,'minimum_driver_free_bytes':a['minimum_driver_free_bytes']})
   if(stage,rank)==(3,0):tail=a;assert a['normal_HTTP_shutdown']and not a['cleanup']['errors']and not a['cleanup']['forced_cleanup']
 supervisorpath=R/'results'/('GLM-latest-full-owner-r290-supervisor-'+LABEL+'.json');supervisor=load(supervisorpath)
 assert supervisor['completed']and supervisor['passed']and supervisor['actual_GPU_handle']==ACTUAL
 assert supervisor['planned_HTTP']==supervisor['planned_Native']==supervisor['attempted_HTTP']==supervisor['completed_HTTP']==supervisor['actual_HTTP']==supervisor['actual_Native']==18
 phase=supervisor['phases']['paired'];assert phase['exit_code']==0;client=phase['result'];cases=client['cases']
 assert client['completed']and client['passed']and client['planned_HTTP']==client['planned_Native']==client['attempted_HTTP']==client['completed_HTTP']==client['actual_HTTP_requests']==len(cases)==18
 clientdir=R/'results'/('GLM-latest-full-owner-r290-client-'+LABEL)/'paired';assert sha(clientdir/'result.json')==phase['result_SHA256']
 sys.path.insert(0,str(S));from workload_r349 import validate_reply
 fixture_results=[]
 for fi,f in enumerate(fixtures):
  rawpath=pathlib.Path(f['request_external_path']);assert sha(rawpath)==f['request_SHA256'];body=load(rawpath)
  if f['id']=='short-read':expected={'path':'input.json'}
  else:
   data=json.loads(next(m['content']for m in body['messages']if m['role']=='tool'and'records'in m['content']));v={r['id']:r['value']for r in data['records']};a,b,c,d=[v[k]for k in data['query_ids']]
   expected={'a':a,'b':b}if f['id']=='anchor-calculate'else{'path':'result.json','content':{'nonce':data['nonce'],'first_sum':a+b,'second_sum':c+d}}
  arm_rows=[]
  for ai in range(6):
   i=fi*6+ai;c=cases[i];r=tail['service_requests'][i];wirepath=clientdir/('response-%02d.wire'%i);wire=load(wirepath)
   assert sha(wirepath)==c['wire_SHA256'];usage=validate_reply(wire,f,json.loads(json.dumps(expected)))
   assert wire['id']==c['request_id']==r['request_id']and usage['completion_tokens']==r['actual_generated_tokens']==r['generated_tokens']and c['semantic_gold_passed']and c['natural_EOS']and c['passed']
   assert c['request_SHA256']==f['request_SHA256']and c['prompt_IDs_SHA256']==f['current_prepared_prompt_IDs_SHA256']
   assert all(c[k]==r[k]for k in ['fixture_id','fixture_index','arm_index','global_index'])
   arm_rows.append({'global_index':i,'arm_index':ai,'window':c['window'],'warmup':c['warmup'],'request_id':wire['id'],'prompt_tokens':usage['prompt_tokens'],'generated_tokens':usage['completion_tokens'],'wire_SHA256':sha(wirepath),'HTTP_wall_s':c['HTTP_wall_s'],'Native_request_wall_s':r['wall_s'],'Native_prefill_CAPI_host_s':r['Native_forward_host_summary']['prefill_s'],'Native_decode_CAPI_host_s':r['Native_forward_host_summary']['decode_s'],'generated_token_ids_SHA256':r['generated_token_ids_SHA256'],'backend_emitter_metrics':c['backend'],'semantic_gold_passed':True,'natural_EOS':True})
  selected=tail['service_requests'][fi*6:fi*6+6];fixture_results.append({'fixture':f['id'],'request_SHA256':f['request_SHA256'],'prompt_IDs_SHA256':f['current_prepared_prompt_IDs_SHA256'],'prompt_tokens':f['current_prepared_prompt_tokens'],'scope':'Unchanged one-chunk control'if fi==0 else'Cross-residency old anchor control'if fi==1 else'Different complete long write_file workload','arms':arm_rows,'measured_comparisons':means(cases[fi*6:fi*6+6],selected),'measured_output_IDs_equal':len({r['generated_token_ids_SHA256']for r in selected[2:]})==1,'warmups_excluded':2})
 oldowner=R/'results/GLM-latest-full-owner-r290-GLM-prefill-W2-W4-Model-r308-stage3-rank0.json';oldsuper=R/'results/GLM-latest-full-owner-r290-supervisor-GLM-prefill-W2-W4-Model-r308.json';o=load(oldowner);oc=load(oldsuper)['phases']['paired']['result']['cases']
 assert o['actual_producer']=='910376.823'and o['library_sha256']==NATIVE and len(o['service_requests'])==6 and all(r['actual_prompt_token_ids_SHA256']==fixtures[1]['current_prepared_prompt_IDs_SHA256']for r in o['service_requests'])
 old_metrics=means(oc,o['service_requests']);new_metrics=fixture_results[1]['measured_comparisons'];residency_control={k:{'old823_W2_mean':v['W2_mean'],'new828_W2_mean':new_metrics[k]['W2_mean'],'new_over_old_W2':new_metrics[k]['W2_mean']/v['W2_mean'],'old823_W4_mean':v['W4_mean'],'new828_W4_mean':new_metrics[k]['W4_mean'],'new_over_old_W4':new_metrics[k]['W4_mean']/v['W4_mean']}for k,v in old_metrics.items()}
 logs=[]
 for stage in range(4):
  p=R/'logs'/(LABEL+'-'+ACTUAL+'-'+str(stage)+'.log');text=p.read_text(errors='replace');assert text.count('GLM_REGULAR_SERVICE_R204_NORMAL '+str(stage))==1 and'Using network IBext_v8'in text and'Using network Socket'not in text;logs.append({'stage':stage,'path':str(p),'SHA256':sha(p),'normal_marker_count':1,'IBext_v8':True,'Socket_fallback':False})
 span=tail['service_requests'][-1]['ended_monotonic']-tail['service_requests'][0]['admission']['accepted_monotonic'];visible=sum(c['backend']['backend_emitter_visible_tokens']for c in cases);assert span>0 and visible>0
 q.update(passed=True,validity='PASS_18_HTTP_NATIVE_32OWNER_SEMANTICS_EOS_NORMAL_DESTROY_POST_CLEAN',actual_HTTP=18,actual_Native=18,actual_owner_count=32,configuration={'TP':8,'PP':4,'cards':32,'cache_rows':1,'cache_capacity':32768,'resident_Model_count':1,'Native_microbatch':1,'HTTP_requests_serial':True,'ordinary_output_budget':4096,'numerical_variant':tail['numerical_variant'],'actual_SDK_SHA256':tail['actual_sdk_sha256'],'W2_W4_meaning':'Prefill control-group window2 versus4; unchanged quantization and decode math'},all32_per_request_input_output_EOS_signatures_equal=True,minimum_driver_free_bytes=min(minimum),fixtures=fixture_results,cross_residency_anchor_control=residency_control,old823_owner_SHA256=sha(oldowner),old823_supervisor_SHA256=sha(oldsuper),owners=owners,network_and_normal=logs,watch_SHA256=sha(watchpath),supervisor_SHA256=sha(supervisorpath),Slurm_terminal_accounting=account,fresh_before=w['fresh_before'],post=w['post'],timer_scope='HTTP full client completion; Native request outer wall; emitter Native CAPI host prefill/decode sums. No sum across pipeline stages.',first_two_warmups_per_fixture_excluded=True,serial18_backend_emitter_throughput={'scope':'All18 including warmups; admission-to-last-completion on the same emitter clock; excludes load; EOS/pad excluded; no onlineSLO qualification','visible_tokens':visible,'interval_s':span,'total_tok_s':visible/span,'per_all32_service_cards_tok_s':visible/span/32,'SLO_qualified_throughput':None})
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
finally:q['completed']=True;print(json.dumps(q))
