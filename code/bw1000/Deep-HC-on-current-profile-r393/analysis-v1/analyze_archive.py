"""Stdlib reader for original current-HC_ON scalar packets and six HTTP records.
Raw archives remain R/local. No Model speedup is inferred from profiler ratios.
"""
from pathlib import Path
import json,hashlib,math,statistics,importlib.util,argparse
D=Path(__file__).resolve().parent;SOURCE=D.parent/'source-v1';NATIVE='976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1'
sha=lambda b:hashlib.sha256(b).hexdigest()
def digest(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':')).encode())
def pair(values):
 off=[values[i]for i in [2,5]];on=[values[i]for i in [3,4]];assert all(math.isfinite(v)and v>0 for v in off+on)
 return dict(OFF=off,ON=on,ON_over_OFF=statistics.mean(on)/statistics.mean(off),OFF_endpoint_drift=off[1]/off[0]-1,ON_endpoint_drift=on[1]/on[0]-1)
def percentile(values,p):
 values=sorted(values);x=(len(values)-1)*p;i=int(x);return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)
def analyze(a,allow_quality_failure=False):
 assert a['schema']==1 and a['step']=='910374.1320';paths=a['paths'];files=a['raw_files']
 for path,f in files.items():assert len(f['text'].encode())==f['bytes']and sha(f['text'].encode())==f['SHA256'],path
 def raw(role):return files[paths[role]]['text'].encode()
 def load(role):return json.loads(raw(role))
 r,c,f,w,legacy=[load(k)for k in ['report','client','fixture','watch','legacy']]
 assert w['publication_commit']=='ca32a72cc7cd03be5f25f594ad7dced2ff44c7d3'
 assert sha(raw('execution_claim'))=='0eedcd52fc109c20fc9eb8b76fb6e7324cd2ad9f5c9c4d6b5be6ce285f98a505';go=load('runtime_GO');assert go['publication_commit']==w['publication_commit']and go['GPU_authorized']and go['root_consumed_final_GPU_scope']
 assert sha(raw('config'))=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09';cfg=load('config')
 assert sha(raw('CPU_whole'))=='831b79259953c81f19cf6981e711558fef9da2457d6a24e68414a6bb38f884a6'and sha(raw('CPU_entry'))=='a042c86cb93585fc5e8b0b96dffcef2a2992881c99ea123188f5171392cac409'
 failed=not(r['completed']and r['passed']);assert r['Native_SHA256']==NATIVE and r['normal_server_shutdown']and len(r['requests'])==6
 if failed:
  assert allow_quality_failure and c['completed']and not c['passed']and c.get('error')=="AssertionError('Actual complete tool task answer differs')"
  assert r.get('normal_Native_shutdown')is False and legacy['engine_worker_shutdown_completed']is True
 else:assert r['normal_Native_shutdown']and r['actual_HTTP']==r['actual_Native']==6
 assert r['source_manifest_SHA256']=='40711240c96ee1e6ca222caedd58a159581d31cf5a03e84511ce4b6db1554671'
 for role,expected in [('owner_manifest','40711240c96ee1e6ca222caedd58a159581d31cf5a03e84511ce4b6db1554671'),('source_manifest','812a3665c25436353a57a05420b1561c3fa4b4d078cdc845fe9209391af74da3'),('controller_manifest','b7c3126fbd0f3104998b6591fddabe4a396482836945cc3f3f6608d82aeb2e1f')]:assert sha(raw(role))==expected
 assert c['completed']and c['actual_HTTP_attempts']==c['actual_HTTP_completed']==c['actual_Native_completed']==6
 assert len(c['requests'])==(5 if failed else 6)and len(paths['wires'])==6 and all(v[:2]==[86,86]for v in r['HC_actual_loaded_modules'])
 assert w['completed']and w['Model_qualified']==(not failed)and w['srun_exit_code']==(1 if failed else 0)and w['terminal'][2:4]==(['FAILED','1:0']if failed else['COMPLETED','0:0'])
 assert w['post_empty']and not w['post']['kfd_pids']and w['post']['HTTP43882_bindable']
 post=a['independent_post'];assert not post['owned_or_old_Deep_processes']and not post['kfd_pids']and post['HTTP43882_bindable']and not post['GPU_imported']
 if not failed:
  owners=r['owner_exit_observation']['owners'];assert len(owners)==len({v['tid']for v in owners})==8 and {v['device']for v in owners}==set(range(8))and not any(v['task_exists_after_destroy']for v in owners)
 assert not r['cleanup']['forced_cleanup']and not r['cleanup']['errors']and not r['cleanup']['HTTP_thread_alive_after_cleanup']
 providers=r['provider_preweight']['actual_new_Native_providers'];required=['deep_owner_arm_r8900','deep_owner_read_r8900','deep_attention_inner_arm_r9193','deep_attention_inner_read_r9193','deep_hc_model_epilogue_set_r366','deep_hc_model_epilogue_get_r366','deep_hc_model_epilogue_counters_r366']
 assert all(providers[k]['SHA256']==NATIVE for k in required)
 calls={v['id']:json.loads(v['function']['arguments'])['path']for m in f['body']['messages']for v in m.get('tool_calls',[])};gold={}
 for m in f['body']['messages']:
  if m['role']=='tool':
   data=json.loads(m['content']);values={v['id']:v['value']for v in data['records']};gold[calls[m['tool_call_id']]]=dict(nonce=data['nonce'],sum=sum(values[i]for i in data['query_ids']))
 gold['total']=sum(v['sum']for v in gold.values());assert gold==f['expected_answer']
 parser=SOURCE/'owner/deep_owner_consumer_r168a.py';mf=load('owner_manifest');assert sha(parser.read_bytes())==mf['files_SHA256'][parser.name]
 spec=importlib.util.spec_from_file_location('pinned393consumer',parser);consumer=importlib.util.module_from_spec(spec);spec.loader.exec_module(consumer)
 rows=[];windows=[];last=None;output_ids=[];clients={v['index']:v for v in c['requests']}
 for i,(native,wirepaths)in enumerate(zip(r['requests'],paths['wires'])):
  client=clients.get(i);assert native['admitted_index']==i and native['HC_mode']==native['prefill_binding']['hc_epilogue_mode']==1
  assert native['prefill_binding']['prefill_chunk']==1024 and native['prefill_binding']['observer_enabled']==(i in [3,4])
  if client:assert client['index']==i and client['hc_epilogue_mode']==1 and client['chunk']==1024
  assert native['normal_reset_before']and native['normal_reset_after']and native['prompt_tokens']==f['actual_prompt_tokens']==8489 and native['max_new_tokens']==128
  assert native['admission']['history_render_audit']['Native_prompt_token_ids_SHA256']==f['actual_prompt_token_ids_SHA256']
  ids=native['generated_token_ids'];assert len(ids)==native['actual_generated_tokens']<=128 and ids[-1]==1 and native['native_finish_reason']=='eos_token';output_ids.append(ids)
  req=json.loads(files[wirepaths['request']]['text']);resp=json.loads(files[wirepaths['response']]['text']);choice=resp['choices'][0]
  assert req['messages']==f['body']['messages']and req['tools']==f['body']['tools']and req['max_tokens']==128 and resp['id']==native['request_id']and choice['finish_reason']=='stop'
  answer=choice['message']['content'].strip()
  if answer.startswith('```json\n')and answer.endswith('\n```'):answer=answer[8:-4]
  correct=json.loads(answer)==gold;assert correct==(not failed or i<5)
  if client:assert files[wirepaths['response']]['SHA256']==client['wire_SHA256']
  trace=[t for t in native['Native_continuation']if t['kind']=='prefill'];past=0
  for t in trace:assert t['past']==past and t['tokens']==min(1024,8489-past);past+=t['tokens'];assert t['total']==past
  assert past==8489 and len(trace)==9;summary=native['Native_forward_host_summary'];assert summary['prefill_calls']==9 and summary['decode_calls']==len(ids)-1
  tokens=[v['tokens']for v in trace]+[1]*(len(ids)-1);coverage=native['HC_forward_coverage'];assert len(coverage)==len(tokens);total=[0]*7
  for v,t in zip(coverage,tokens):
   delta=[0,0,86,86 if t==1 else 0,86 if t==1 else 0,86 if t!=1 else 0,0];assert v['input_tokens']==t and v['mode']==1 and v['per_owner_delta']==[delta]*8 and v['candidate_completed_branch']==(t==1);total=[a+b for a,b in zip(total,delta)]
  before,after=native['HC_counters_before_request'],native['HC_counters_after_request'];assert last is None or before==last;assert all([b-a for a,b in zip(x,y)]==total for x,y in zip(before,after));last=after
  times=native['forward_timings'];assert len(times)==len(tokens)and len([v for v in times if v['kind']=='decode'])==len(ids)-1
  for call,v in enumerate([v for v in times if v['kind']=='decode'],1):assert v['decode_call_1based']==call and v['past']==8489+call-1 and v['tokens']==1 and v['observer_enabled']==(i in [3,4]and call in [8,87])
  profiles=native.get('owner_profiles',[]);assert [v['geometry']['decode_call_1based']for v in profiles]==([8,87]if i in [3,4]else[])
  for v in profiles:
   assert v['native_window_closed']and v['owner_event_validation_passed']and v['inner_event_validation_passed'];packets={}
   for kind,meta in v['scalar_packets'].items():
    source=next(x['host_path']for x in paths['profile_packets']if x['container_path']==meta['path']);assert files[source]['SHA256']==meta['SHA256'];packets[kind]=files[source]['text'].encode()
   outer=consumer.validate_owner(packets['owner'],1);innerraw=json.loads(packets['inner']);kinds=consumer.native_expected_kinds(cfg)
   inner=consumer.validate_inner(packets['inner'],kinds,1,1);bindings={(x['device'],x['layer']):(x['owner'],x['stream'])for x in outer['raw_owner_events']['rows']};assert all(bindings[(x['device'],x['layer'])]==(x['owner'],x['stream'])for x in innerraw['rows'])
   for sourcekey,key in [('old_perstage_perdevice_ms','perstage_perdevice_sum_layers_stream_elapsed_ms'),('old_host_wall_perstage_perdevice_ms','perstage_perdevice_sum_layers_host_wall_ms'),('old_thread_cpu_perstage_perdevice_ms','perstage_perdevice_sum_layers_thread_cpu_ms')]:assert v[sourcekey]==outer[key]
   assert v['inner_persegment_perdevice_ms']==inner['persegment_perdevice_sum_layers_stream_elapsed_ms']and v['inner_total_perdevice_ms']==inner['all_sibling_segments_total_perdevice_ms']
   groups={str(k):{name:[0.0]*8 for name in consumer.INNER_SEGMENTS}for k in [0,1,2]}
   drains={'decoder':[0.0]*8,'attention':[0.0]*8}
   for x in outer['raw_owner_events']['rows']:drains['decoder'][x['device']]+=x['drain_host_ms']
   for x in innerraw['rows']:
    drains['attention'][x['device']]+=x['original_final_drain_host_ms']
    for seg in x['segments']:groups[str(x['kind'])][consumer.INNER_SEGMENTS[seg['segment']]][x['device']]+=seg['stream_ms']
   windows.append(dict(request_index=i,decode_call=v['geometry']['decode_call_1based'],past=v['geometry']['past'],current_HC_ON_perstage_perowner_ms=v['old_perstage_perdevice_ms'],current_HC_ON_hostwall_perstage_perowner_ms=v['old_host_wall_perstage_perdevice_ms'],current_HC_ON_threadCPU_perstage_perowner_ms=v['old_thread_cpu_perstage_perdevice_ms'],inner_siblings_perowner_ms=inner['persegment_perdevice_sum_layers_stream_elapsed_ms'],kind_segments_perowner_ms=groups,final_drain_perowner_ms=drains,raw_packets=v['scalar_packets']))
  ticks=[t for t,token in zip(native['backend_token_monotonic'],ids)if token!=1];assert len(ticks)==len(ids)-1 and all(math.isfinite(t)for t in ticks)and all(b>a for a,b in zip(ticks,ticks[1:]));rate=(len(ticks)-1)/(ticks[-1]-ticks[0])
  if client:assert math.isclose(rate,client['backend_emitter_decode_tok_s'],rel_tol=1e-12)
  rows.append(dict(index=i,warmup=i<2,HC_mode=1,observer=i in [3,4],input_tokens=8489,output_including_EOS=len(ids),visible=len(ticks),gold=correct,EOS=True,output_ID_SHA256=digest(ids),HTTP_wall_s=client['client_HTTP_wall_s']if client else None,HTTP_wall_missing_reason=None if client else'Goldfailedbeforeclientrowappend; no persistedlastHTTPelapsed. UNKNOWN, notinferred',Native_summary_prefill_s=summary['prefill_s'],Native_summary_decode_s=summary['decode_s'],exact_callback_prefill_s=sum(v['native_forward_wrapper_s']for v in times if v['kind']=='prefill'),exact_callback_decode_s=sum(v['native_forward_wrapper_s']for v in times if v['kind']=='decode'),HC_counter_observation_s=native['HC_counter_observation_s'],observer_inclusive_sum_s=sum(v['observer_total_wall_s']for v in profiles),backend_visible_decode_tok_s=rate,backend_TTFT_s=ticks[0]-native['admission']['accepted_monotonic'],ITL_p95_s=percentile([b-a for a,b in zip(ticks,ticks[1:])],.95),ITL_p99_s=percentile([b-a for a,b in zip(ticks,ticks[1:])],.99),neighbor_timings=[v for v in times if v['kind']=='decode'and v['decode_call_1based']in [7,8,9,86,87,88]],memory_free_before_all8_bytes=[v[0]for v in native['GPU_free_before_request_all8']],memory_free_after_all8_bytes=[v[0]for v in native['GPU_free_all8']]))
 sampled={}
 for index in [8,87]:
  values=[next(v['native_forward_wrapper_s']for v in x['forward_timings']if v['kind']=='decode'and v['decode_call_1based']==index)for x in r['requests']];sampled[str(index)]=pair(values);sampled[str(index)]['formal_input_generated_prefix_IDs_equal']=len({digest(output_ids[i][:index])for i in [2,3,4,5]})==1
 pairs={key:pair([v[key]for v in rows])if all(v[key]is not None for v in rows)else None for key in ['HTTP_wall_s','Native_summary_prefill_s','Native_summary_decode_s','exact_callback_prefill_s','exact_callback_decode_s','backend_visible_decode_tok_s']}
 geometry={}
 for i in [2,5]:
  for v in r['requests'][i]['forward_timings']:
   if v['kind']=='decode':
    key='C4_C128_completion'if(v['past']+1)%128==0 else'C4_only_completion'if(v['past']+1)%4==0 else'noncompletion';geometry.setdefault(key,[]).append(v['native_forward_wrapper_s'])
 return dict(passed=not failed,diagnostic_data_checks_passed=True,functional_qualification='FAIL_5_OF6'if failed else'PASS_6_OF6',actual_GPU=a['step'],CPU_keeper=w['CPU_keeper_actual'],Native_SHA256=NATIVE,HC_mode_fixed1=True,source_publication_commit=w['publication_commit'],rows=rows,sampled_observer_perturbation=sampled,full_pairs=pairs,current_HC_ON_instrumented_windows=windows,formal_OFF_geometry={k:dict(calls=len(v),mean_callback_s=statistics.mean(v),sum_callback_s=sum(v),max_callback_s=max(v))for k,v in geometry.items()},loading_s=legacy['loading_seconds'],all8normal=not failed,lifetime_evidence=dict(original_finally_engine_shutdown_completed=legacy.get('engine_worker_shutdown_completed'),report_normal_Native_shutdown=r.get('normal_Native_shutdown'),normal_success_poststamp_absent=failed,whole_lifecycle_PASS=not failed),post=post,actual_Slurm=a['accounting'],evidence_manifest={k:dict(SHA256=v['SHA256'],bytes=v['bytes'])for k,v in files.items()},performance_gain=False,scope='Current976f HC_ON directeddiagnosis only; actualqualityFAIL preserved. Fourwindows not fullcorrect-request qualification. Nooptimizationchange/Modelgain/speedupclaim.',limits=['Four instrumented windows do not representall97decodecalls.','Owner16ranges nested, inner6siblings sequential; do not addacrossowners/nestedranges. Eventintervals includehostgaps/dependencies/comm/sync, notGPUbusy orNIC-only.','Native_summary includesSparseObserver onselectedcalls; exact_callback excludesobserverarm/read/packetIO. observer_total is inclusive andnotadditive. HCcounters separate.','Generatedprefix mismatch makes perturbation ratio confounded bytrajectory, notobservercausaloverhead. Sameprefix ratios stillincludeendpoint drift.','Memorycheckpoints notpeak; allocatedwall/cardseconds notGPUbusy. BufferedHTTP notclientincrementalTTFT/SLO.','Prior353HC_OFFnotcurrentHC_ON; no crossNative timingcomparison orphysicalpeak claim.','Gold5/6 FAIL. LastHTTPwall absent, HTTPpairUNKNOWN. Originalfinallyshutdowntrue doesnot fabricate normalNativepostsuccessstamp.'])
def main():
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output');p.add_argument('--allow-quality-failure',action='store_true');a=p.parse_args();q=json.loads(Path(a.archive).read_bytes());q=q.get('archive',q);result=analyze(q,a.allow_quality_failure);text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
 if a.output:Path(a.output).write_text(text,encoding='utf-8',newline='\n')
 else:print(text)
if __name__=='__main__':main()
