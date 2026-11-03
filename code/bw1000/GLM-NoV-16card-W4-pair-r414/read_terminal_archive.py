"""Independent stdlib-only reader of sealed853 original bytes. Never imports runtime."""
from pathlib import Path
import argparse,hashlib,json,math,statistics
NATIVE='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
OWNER='5e944fb676f15022f1387d936cc9286c00953f0ca6884ef79b12979b4bca8ac4'
EOS={154820,154827,154829}
digest=lambda b:hashlib.sha256(b).hexdigest()
def pairs(items):
 d={}
 for k,v in items:
  if k in d:raise ValueError('Duplicate JSON key:'+k)
  d[k]=v
 return d
def loads(text):
 def bad(v):raise ValueError('Nonfinite JSON constant:'+v)
 return json.loads(text,object_pairs_hook=pairs,parse_constant=bad)
def percentile(values,p):
 if not values:return None
 v=sorted(values);x=(len(v)-1)*p;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def exact_values(actual,expected):
 assert type(actual)is type(expected),(type(actual).__name__,type(expected).__name__)
 if isinstance(expected,dict):assert actual.keys()==expected.keys();[exact_values(actual[k],v)for k,v in expected.items()]
 elif isinstance(expected,list):assert len(actual)==len(expected);[exact_values(a,b)for a,b in zip(actual,expected)]
 else:assert actual==expected,(actual,expected)
def read(path):
 raw=Path(path).read_bytes();outer=loads(raw);a=outer.get('archive',outer)
 if 'archive' in outer:
  assert digest((json.dumps(a,ensure_ascii=False,indent=2)+'\n').encode())==outer['archive_SHA256']
 assert a['actual_GPU']=='910376.853'and a['actual_keeper']=='910407.833'and a['run']=='pair-20261103-a'
 f=a['files'];roles=a['roles'];assert len(f)==a['file_count']
 for p,e in f.items():
  b=e['rawtext'].encode(e['encoding']);assert p==e['path']and len(b)==e['bytes']and digest(b)==e['SHA256'],p
 def entry(role):return f[roles[role]]
 def obj(role):return loads(entry(role)['rawtext'])
 m=obj('GPU_manifest');assert entry('GPU_manifest')['SHA256']=='3a9cbe1e68706369ba8bdd319ec574d52812e007e20dc7a0f9008a58c400fdd8'
 assert m['Native_SHA256']==NATIVE and m['owner_manifest_SHA256']==OWNER and(m['TP'],m['PP'],m['rows'],m['context'])==(8,2,1,32768)
 for role,prefix in [('served_owner_manifest','served_source:'),('root_source_manifest','ancestry_source:')]:
  manifest=obj(role)
  for n,h in manifest['files_SHA256'].items():assert entry(prefix+n)['SHA256']==h,n
 for n,h in m['additional_files_SHA256'].items():assert entry('entry_source:'+n)['SHA256']==h,n
 assert entry('Native_CPU804')['SHA256']==m['Native_CPU804_receipt_SHA256']and entry('CPU832')['SHA256']==m['CPU_receipt_SHA256']and entry('CPU820')['SHA256']==m['CPU820_receipt_SHA256']
 assert obj('Native_CPU804')['library_SHA256']==NATIVE and a['binary_references'][0]['SHA256']==NATIVE
 watch=obj('watch');assert watch['completed']and watch['passed']and watch['actual_GPU']==a['actual_GPU']and watch['CPU_keeper']==a['actual_keeper']and not watch['forced_cleanup']and watch['srun_exit_code']==0
 byid={r['JobIDRaw']:r for r in a['accounting']}
 for handle in ['910376.853','910407.833']+list({v['actual_step']for v in watch['fresh_before']+watch['post']}):assert byid[handle]['State']=='COMPLETED'and byid[handle]['ExitCode']=='0:0'
 assert len(watch['post'])==2 and all(not x['kfd_pids']and not x['matching_component_processes']and not x['expected_PIDs_still_exist']and x['ports_bindable']=={'43843':True}and x['TCP_connections_made']==0 for x in watch['post'])
 source0=entry('served_source:owner_r290.py')['rawtext'];assert "free>=512*1024**2"in source0 and "torch.isfinite(logits).all()"in source0
 assert "8 * 1024 ** 3"in entry('served_source:model_entry_r290.py')['rawtext']
 owners=[];signatures=[];memory=[];source_calls=[]
 for stage in range(2):
  for rank in range(8):
   tag='s%d-r%d'%(stage,rank);q=obj('owner:'+tag);l=obj('lifecycle:'+tag);c=obj('constructor:'+tag)
   assert q['stage']==stage and q['rank']==rank and q['actual_producer']=='910376.853'and q['source_manifest_SHA256']==OWNER and q['library_sha256']==NATIVE
   assert all(q[k]is True for k in ['completed','worker_shutdown_completed','post_close_Torch_liveness','normal_reset_close','native_service_loop_completed','normal_shutdown','constructor_noV_verified','normal_reset'])
   assert q['context_policy']=='v403-no-resident-V'and q['context_policy_env']=='1'and(q['TP'],q['PP'],q['all_serving_cards'],q['cache_rows'],q['cache_capacity'])==(8,2,16,1,32768)
   assert q['actual_r403_Native_binding']['library_SHA256']==NATIVE and len(q['actual_r403_Native_binding']['CAPI_providers'])==8
   assert l['completed']and l['normal']and not l['primary_error']and not l['cleanup_errors']and l['destroy_attempts']==l['shutdown_attempts']==1 and l['destroy_RC']==l['shutdown_RC']==0
   assert l['existing_context_only']and l['constructor_calls_expected']==q['constructor_calls_expected']==1 and l['source_manifest_SHA256']==OWNER and l['actual_step']=='910376.853'
   assert c['Native_SHA256']==NATIVE and c['source_manifest_SHA256']==OWNER and c['actual_step']=='910376.853'and not c['loaded']and entry('constructor:'+tag)['SHA256']==q['constructor_snapshot_SHA256']
   census=c['census'];assert(census['rows'],census['capacity'],census['layer_begin'],census['layer_end'])==(1,32768,[0,38][stage],[38,78][stage+0])
   assert census['cache_category_bytes'].get('projected_value',0)==0 and sum(census['cache_category_bytes'].values())==census['cache_unique_start_bytes']
   assert len(census['cache_layers'])==[38,40][stage]
   for layer in census['cache_layers']:
    kinds={t['category']for t in layer['tensors']};assert 'projected_value'not in kinds and {'latent','rotary','projected_key'}<=kinds
   assert q['actual_partition']['begin']==[0,38][stage]and q['actual_partition']['end']==[38,78][stage]
   assert q['registry_owner_before_weightIO']['all16passed']and len(q['registry_owner_before_weightIO']['identities']['identities'])==[38,40][stage]
   assert q['memory_after_create'][0]>=8*(1<<30)and q['minimum_driver_free_bytes']>=512*(1<<20)
   phases=[loads(line)for line in entry('phase:'+tag)['rawtext'].splitlines()if line]
   for name in ['create.enter','create.return','load.enter','load.return','raw_weight_load.return.before_finish','finish.enter','finish.return']:assert sum(x['phase']==name for x in phases)==1
   assert all(x['stage']==stage and x['rank']==rank and x['PID']==q['PID']and x['is_peak']is False for x in phases)
   pt={x['phase']:x for x in phases};duration=lambda x,y:(pt[y]['driver_monotonic_ns']-pt[x]['driver_monotonic_ns'])/1e9
   assert len(q['service_requests'])==q['full_requests_completed']==q['planned_Native']==q['attempted_Native']==q['completed_Native']==6
   sig=[];timings=[]
   for i,r in enumerate(q['service_requests']):
    assert(r['fixture_id'],r['prompt_tokens'],r['pair_index'],r['prefill_control_window'],r['requested_output_budget'])==('long-write',9741,i,[2,4,2,4,4,2][i],4096)
    assert r['paired_warmup']==(i<2)and r['original_reset_all16_completed']and r['generated_natural_EOS']and r['reason']=='eos_token'
    calls=r['native_calls'];pre=[x for x in calls if x['prefill']];dec=[x for x in calls if not x['prefill']]
    assert len(pre)==20 and sum(x['T']for x in pre)==r['prompt_tokens']and all(x['T']==1 for x in dec)and len(dec)==r['generated_tokens']-1
    assert all(math.isfinite(x['Native_CAPI_s'])and x['Native_CAPI_s']>=0 for x in calls)
    sig.append([r['actual_prompt_token_ids_SHA256'],r['generated_token_ids_SHA256'],r['generated_tokens'],r['reason']])
    timings.append(dict(fixture_id=r['fixture_id'],owner_request_wall_s=r['wall_s'],Native_CAPI_prefill_s=sum(x['Native_CAPI_s']for x in pre),Native_CAPI_decode_s=sum(x['Native_CAPI_s']for x in dec),prefill_calls=len(pre),decode_calls=len(dec)))
   signatures.append(sig);source_calls.append(timings)
   memory.append(dict(stage=stage,rank=rank,weight_logical_bytes=census['weight_unique_start_bytes'],cache_logical_bytes=census['cache_unique_start_bytes'],cache_category_bytes=census['cache_category_bytes'],before_create_free_bytes=pt['create.enter']['free'],after_create_free_bytes=pt['create.return']['free'],preweight_after_warm_free_bytes=q['memory_after_create'][0],after_loading_free_bytes=q['free_total_gpu_bytes_after_loading'][0],minimum_sampled_forward_driver_free_bytes=q['minimum_driver_free_bytes'],phase_snapshots=phases,Native_allocator_peak='UNKNOWN',snapshots_are_peak=False))
   owners.append(dict(stage=stage,rank=rank,owner_SHA256=entry('owner:'+tag)['SHA256'],lifecycle_SHA256=entry('lifecycle:'+tag)['SHA256'],constructor_SHA256=entry('constructor:'+tag)['SHA256'],constructor_phase_pairs=1,destroy_attempts=l['destroy_attempts'],shutdown_attempts=l['shutdown_attempts'],create_wall_s=duration('create.enter','create.return'),raw_load_wall_s=duration('load.enter','raw_weight_load.return.before_finish'),finish_wall_s=duration('finish.enter','finish.return'),load_and_finish_wall_s=duration('load.enter','load.return'),actual_owner_weight_load_wall_seconds=q['actual_owner_weight_load_wall_seconds'],loading_seconds=q['loading_seconds']))
 assert all(s==signatures[0]for s in signatures)
 emitter=obj('owner:s1-r0');assert emitter['normal_HTTP_shutdown']and emitter['automatic_CPU_client_exit_code']==0 and not emitter['automatic_CPU_client_group_forced_stop']and not emitter['cleanup']['forced_cleanup']and not emitter['cleanup']['errors']
 client=obj('client_result');supervisor=obj('supervisor');assert client['completed']and client['passed']and client['attempted_HTTP']==client['completed_HTTP']==client['actual_HTTP_requests']==6
 assert supervisor['completed']and supervisor['passed']and supervisor['actual_GPU_handle']=='910376.853'and supervisor['actual_HTTP']==supervisor['actual_Native']==6
 card=obj('fixture_card');requests=[]
 for i,r in enumerate(emitter['service_requests']):
  fixture=card['fixtures'][0];body=obj('request:'+str(i));assert entry('request:'+str(i))['SHA256']==fixture['request_SHA256']
  assert body['max_tokens']==4096 and body['stream']is False and body['temperature']==0 and body['parallel_tool_calls']is False and body['chat_template_kwargs']=={'clear_thinking':False}
  values=loads(next(x['content']for x in body['messages']if x['role']=='tool'and 'records'in x['content']));records=values['records'];v={x['id']:x['value']for x in records};assert len(v)==len(records);ints=[v[k]for k in values['query_ids']];assert len(ints)==4 and all(type(x)is int for x in ints)
  expected={'path':'result.json','content':{'nonce':values['nonce'],'first_sum':ints[0]+ints[1],'second_sum':ints[2]+ints[3]}}
  wire=loads(entry('response:'+str(i))['rawtext']);choice=wire['choices'];assert len(choice)==1 and choice[0]['finish_reason']=='tool_calls'
  tools=choice[0]['message']['tool_calls'];assert len(tools)==1 and tools[0]['type']=='function'and tools[0]['function']['name']=='write_file'
  arguments=loads(tools[0]['function']['arguments'])
  assert isinstance(arguments['content'],str);arguments['content']=loads(arguments['content'])
  exact_values(arguments,expected);assert wire['id']==r['request_id']and entry('response:'+str(i))['SHA256']==client['cases'][i]['wire_SHA256']
  prompt=obj('prompt_IDs:'+str(i));ids_in=prompt['prompt_token_ids'];assert len(ids_in)==9741 and digest(json.dumps(ids_in,separators=(',',':')).encode())==r['actual_prompt_token_ids_SHA256']==fixture['current_prepared_prompt_IDs_SHA256']
  ids=r['generated_token_ids'];ticks=r['backend_token_monotonic'];assert len(ids)==len(ticks)==r['actual_generated_tokens']==r['generated_tokens']==wire['usage']['completion_tokens']
  assert all(type(x)is int and 0<=x<154880 for x in ids)and ids[-1]in EOS and not any(x in EOS for x in ids[:-1]);assert digest(json.dumps(ids,separators=(',',':')).encode())==r['generated_token_ids_SHA256']
  assert all(math.isfinite(t)for t in ticks)and all(x<=y for x,y in zip(ticks,ticks[1:]));accepted=r['admission']['accepted_monotonic'];started=r['started_monotonic'];ended=r['ended_monotonic'];assert accepted<=started<=ticks[0]<=ticks[-1]<=ended and r['normal_reset_before']and r['normal_reset_after']and r['native_finish_reason']=='eos_token'
  vis=[t for t,k in zip(ticks,ids)if k not in EOS];gaps=[y-x for x,y in zip(vis,vis[1:])];dec=(len(vis)-1)/(vis[-1]-vis[0]);timing=client['cases'][i]['backend'];assert math.isclose(dec,timing['backend_emitter_decode_tok_s'],rel_tol=1e-10)
  pre=sum(x['Native_CAPI_s']for x in r['native_calls']if x['prefill']);decode=sum(x['Native_CAPI_s']for x in r['native_calls']if not x['prefill']);assert math.isclose(pre,r['Native_forward_host_summary']['prefill_s'],rel_tol=1e-12)and math.isclose(decode,r['Native_forward_host_summary']['decode_s'],rel_tol=1e-12)
  per=[x[i]for x in source_calls];requests.append(dict(fixture_id=fixture['id'],prompt_tokens=len(ids_in),input_body_SHA256=fixture['request_SHA256'],prompt_IDs_SHA256=r['actual_prompt_token_ids_SHA256'],all16_HTTP_window_card_seconds=16*client['cases'][i]['HTTP_wall_s'],ordinary_budget=4096,index=i,warmup=i<2,prefill_window=[2,4,2,4,4,2][i],generated_tokens_including_EOS=len(ids),visible_useful_tokens=len(vis),generated_IDs_SHA256=r['generated_token_ids_SHA256'],natural_EOS=True,semantic_expected=expected,semantic_actual=arguments,semantic_gold_passed=True,all16owner_signature_equal=True,normal_reset=True,HTTP_wall_s=client['cases'][i]['HTTP_wall_s'],backend_TTFT_from_admission_s=vis[0]-accepted,backend_TTFT_from_Native_start_s=vis[0]-started,backend_first_last_s=vis[-1]-vis[0],backend_decode_tok_s=dec,backend_ITL_p50_s=percentile(gaps,.5),backend_ITL_p95_s=percentile(gaps,.95),backend_ITL_p99_s=percentile(gaps,.99),backend_max_ITL_s=max(gaps),backend_request_started_to_ended_s=ended-started,backend_queue_s=started-accepted,emitter_owner_wall_including_reset_s=r['wall_s'],emitter_Native_prefill_s=pre,emitter_Native_decode_s=decode,max16owner_request_wall_s=max(x['owner_request_wall_s']for x in per),max16owner_Native_prefill_s=max(x['Native_CAPI_prefill_s']for x in per),max16owner_Native_decode_s=max(x['Native_CAPI_decode_s']for x in per),all16owner_percall_summaries=per))
 logs=[e for p,e in f.items()if '/logs/'in p and 'pair-20261103-a-910376.853-'in p and p.endswith('.log')];assert len(logs)==2
 assert all('GLM_REGULAR_SERVICE_R204_NORMAL '+str(stage)in next(x['rawtext']for x in logs if x['path'].endswith('-%d.log'%stage))for stage in range(2))
 # Only same-clock per-request durations and ratios. Output length is retained
 # independently, not forced equal or normalized into a claimed speedup.
 metrics=['HTTP_wall_s','backend_TTFT_from_admission_s','backend_first_last_s','backend_decode_tok_s','emitter_Native_prefill_s','emitter_Native_decode_s','max16owner_request_wall_s']
 paired={}
 for key in metrics:
  v=[q[key]for q in requests];off=[v[2],v[5]];on=[v[3],v[4]]
  paired[key]=dict(formal_OFF=off,formal_ON=on,adjacent_ON_over_OFF=[v[3]/v[2],v[4]/v[5]],mean_ON_over_OFF=statistics.mean(on)/statistics.mean(off),OFF_endpoint_ratio=v[5]/v[2],OFF_endpoint_absolute_fractional_drift=abs(v[5]/v[2]-1),direction='higherbetter'if key=='backend_decode_tok_s'else'lowerbetter')
 assert [r['prefill_window']for r in requests]==[2,4,2,4,4,2]
 assert [r['warmup']for r in requests]==[True,True,False,False,False,False]
 assert all(r['input_body_SHA256']==requests[0]['input_body_SHA256']and r['prompt_IDs_SHA256']==requests[0]['prompt_IDs_SHA256']for r in requests)
 finalcli=obj('final_Bash_CLI');assert finalcli['returncode']==0 and finalcli['Slurm_submitted']==0
 cli=loads(finalcli['stdout']);assert cli['preflight_passed']and not cli['intent_created']and cli['shell_contract']['actual_Bash']and len(cli['shell_contract']['roles'])==2
 step_s=int(byid['910376.853']['ElapsedRaw']);http=sum(r['HTTP_wall_s']for r in requests);useful=sum(r['visible_useful_tokens']for r in requests)
 return dict(state='SCOPED_C1_TP8_PP2_NO_V_W2_W4_SIX_REQUESTS_NORMAL_PASS',audit_passed=True,actual_GPU='910376.853',actual_keeper='910407.833',source_publication=a['publication_commit'],Native_SHA256=NATIVE,owner_manifest_SHA256=OWNER,GPU_manifest_SHA256=a['GPU_manifest_SHA256'],sealed_archive_path=str(Path(path).resolve()),sealed_local_file_SHA256=digest(raw),sealed_remote_archive_path=outer.get('archive_path'),sealed_remote_archive_SHA256=outer.get('archive_SHA256'),all_raw_files_bytes_SHA_verified=len(f),actual_HTTP=6,actual_Native_logical_requests=6,warmup_requests=2,formal_indices=[2,3,4,5],paired_formal=paired,variable_generated_lengths=[x['generated_tokens_including_EOS']for x in requests],formal_generated_lengths=[requests[i]['generated_tokens_including_EOS']for i in [2,3,4,5]],normal16owners=True,all16_same_tokens_EOS_reset=True,normal_context_create_phase_pairs=16,normal_destroy_once=16,normal_shutdown_once=16,normal_mp_spawn_Slurm_srun_keeper=True,independent_post=watch['post'],original_8GiB_preweight_admission_all16_passed=True,original512MiB_sampled_perforward_guard_all16_passed=True,finite_guard='Actualsource tail8owner isfinite check on every returned forward completed without exception; rawlogits not retained for independent recomputation.',requests=requests,owners=owners,memory=memory,cost=dict(all_serving_cards=16,GPU_step_wall_s=step_s,allocated_GPU_card_seconds=16*step_s,CPU_keeper_elapsed_s=int(byid['910407.833']['ElapsedRaw']),sum_six_HTTP_wall_s=http,visible_useful_tokens=useful,HTTP_window_visible_tokens_per_s_per_card=useful/http/16,residency_visible_tokens_per_s_per_card=useful/step_s/16,maximum_recorded_same_owner_load_and_finish_wall_s=max(x['load_and_finish_wall_s']for x in owners),GPU_busy_time='UNKNOWN'),limits=['One complete9741 toolhistory repeated six times in a single residency;first2warm excluded from pairing;actualcapacity32768/C1; not a full32768-token generation, benchmark/no-regression or arbitrary-workload peak-memory guarantee.','Original8GiB preweight and512MiB runtime sampled floor unchanged; driver snapshots and sampledminimum are not GPUallocator peak or minimum instantaneous free. Nativeallocator live/reserved decomposition UNKNOWN.','ActualHTTP is buffered untilcompletion. Backendemitter TTFT/ITL are same-emitter monotonic-clock measurements, not publicSSE/onlineSLO; no crossnode clock subtraction.','Native_CAPI includes actual synchronous C++ enclosure but no separate CPP internalphase or GPUevent is retained in this owner schema. Native_CAPI sums, ownerrequestwall, backendelapsed andHTTPwall are nested/overlapping scopes and must not be added. Perowner maxima compare durations, not crossnode timestamp differences.','GeneratedIDs/EOS/hash and all16 signatures verified; reader independently checks actualwire gold but does not independently rerun tokenizer decoding of allgeneratedIDs.','Same16card W2/W4 ABBA, no32card control: no claim16vs32 speedup or >15tok/s. All16cards charged; no pad/EOS counted as useful tokens.'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True);a=p.parse_args();result=read(a.archive);out=Path(a.output);assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(output=str(out),SHA256=digest(out.read_bytes()),audit_passed=True,actual_GPU=result['actual_GPU'],requests=[{k:r[k]for k in ['fixture_id','visible_useful_tokens','HTTP_wall_s','backend_decode_tok_s','semantic_gold_passed']}for r in result['requests']])))
