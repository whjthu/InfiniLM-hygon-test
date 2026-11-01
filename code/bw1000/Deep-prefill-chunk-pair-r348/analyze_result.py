"""Summarize the frozen six-request trial. Never extrapolate decode/SLO gains."""
import pathlib,json,argparse,math,hashlib
def analyze(q):
 result=dict(actual_step=q['actual_step'],Native_SHA256=q['Native_SHA256'],scope='Single same-model residency; one frozen actual8489token tool task; 2warm+oneABBA1024/2048. No widequality/SLO/no-regression proof.',normal_server_shutdown=q.get('normal_server_shutdown'),normal_Native_shutdown=q.get('normal_Native_shutdown'),eight_owner_exit=q.get('owner_exit_observation'),passed=False)
 rows=q['requests'];result['actual_Native_observed']=len(rows)
 if not(q.get('completed')and q.get('passed')and q.get('normal_server_shutdown')and q.get('normal_Native_shutdown')and q.get('owner_exit_observation',{}).get('all8_unique_threads_exited')):
  result.update(outcome='UNQUALIFIED_OR_INCOMPLETE',owner_error=q.get('owner_error'));return result
 assert q.get('actual_HTTP')==q.get('actual_Native')==len(rows)==6
 client=q['phases']['prefill-pair']['result'];assert client['passed']and client['completed']and len(client['requests'])==6
 assert client['planned_HTTP']==client['planned_Native']==client['actual_HTTP_attempts']==client['actual_HTTP_completed']==client['actual_Native_observed']==client['actual_Native_completed']==6
 expected=[1024,2048,1024,2048,2048,1024];out=[];prompt_hashes=[]
 for i,(r,c,chunk)in enumerate(zip(rows,client['requests'],expected)):
  assert r['admitted_index']==i and r['prefill_binding']['prefill_chunk']==chunk and r['max_new_tokens']==128 and r['prompt_tokens']==8489 and r['native_finish_reason']=='eos_token'and r['generated_token_ids'][-1]==1 and r['actual_generated_tokens']<=128
  assert c['task_correct']and c['natural_EOS']and c['chunk']==chunk
  trace=[t for t in r['Native_continuation']if t['kind']=='prefill'];assert len(trace)==(8489+chunk-1)//chunk and sum(t['tokens']for t in trace)==8489 and all(t['tokens']<=chunk for t in trace)
  h=r['admission']['history_render_audit'];assert h['qualified']and h['prequeue_bound']and not h['markers_sent_to_Native'];prompt_hashes.append(h['Native_prompt_token_ids_SHA256'])
  f=r['Native_forward_host_summary'];ticks=[t for t,k in zip(r['backend_token_monotonic'],r['generated_token_ids'])if k!=1];ttft=ticks[0]-r['started_monotonic'];assert math.isfinite(ttft)and ttft>0
  out.append(dict(index=i,chunk=chunk,warmup=i<2,prompt_tokens=8489,prefill_calls=f['prefill_calls'],prefill_native_s=f['prefill_s'],prefill_input_tok_s=8489/f['prefill_s'],prefill_input_tok_s_per_card_using_all8=8489/f['prefill_s']/8,backend_emitter_first_wait_from_Native_start_s=ttft,client_complete_HTTP_s=c['client_HTTP_wall_s'],decode_native_s=f['decode_s'],decode_native_calls=f['decode_calls'],visible_backend_decode_tok_s=c['backend_emitter_decode_tok_s'],generated_including_EOS=r['actual_generated_tokens'],natural_EOS=True,task_correct=True,min_free_before_request_all8_bytes=min(v[0]for v in r['GPU_free_before_request_all8']),min_free_after_request_all8_bytes=min(v[0]for v in r['GPU_free_all8'])))
 assert len(set(prompt_hashes))==1;result.update(passed=True,outcome='COMPLETE_TASK_CORRECT_NATURAL_EOS6_WITH_NORMAL_LIFECYCLE',prompt_token_ids_SHA256=prompt_hashes[0],rows=out,quality_scope='Six repeats of one actual complete tool task only; token equality not required; output128',memory_scope='Before/after physical free checkpoints only; no measured allocator/kernel peak')
 def paired(key):
  a=(out[2][key]+out[5][key])/2;b=(out[3][key]+out[4][key])/2
  return dict(OFF1024_mean=a,ON2048_mean=b,ON_over_OFF=b/a,OFF_endpoint_drift=out[5][key]/out[2][key]-1,B_endpoint_drift=out[4][key]/out[3][key]-1,OFF_endpoints=[out[2][key],out[5][key]],ON_middle=[out[3][key],out[4][key]],paired_groups=1,warmups_excluded=[0,1])
 result['paired']={key:paired(key)for key in ['prefill_native_s','backend_emitter_first_wait_from_Native_start_s','client_complete_HTTP_s','decode_native_s','visible_backend_decode_tok_s']};result['timing_scope']='Native forward-wrapper host wall (same Python argument packaging + actual CAPI + drain), backend emitter first-token wait, full buffered HTTP completion reported separately. No client incremental TTFT/ITL qualification or GPUbusy claim; decode measured but not a chunking-speed hypothesis.';result['not_entire_task_goal']='No claim of reaching strict single-user>15tok/s; no synthetic-model extrapolation.';return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('output');a=p.parse_args();path=pathlib.Path(a.report);out=pathlib.Path(a.output);assert not out.exists();q=analyze(json.loads(path.read_bytes()));q['source_report_SHA256']=hashlib.sha256(path.read_bytes()).hexdigest();out.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
