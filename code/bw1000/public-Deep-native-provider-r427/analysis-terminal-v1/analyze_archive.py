"""Stdlib independent original public request/wire/Native terminal audit; R/L raw only."""
import argparse,hashlib,json,math,re,statistics
from pathlib import Path
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def compact(q):return json.dumps(q,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()
def unique(pairs):
 q={}
 for k,v in pairs:
  if k in q:raise ValueError('Duplicate JSON key')
  q[k]=v
 return q
def parsed_final(text):
 text=text.strip();presentation='bare_JSON'
 if text.startswith('```'):
  m=re.fullmatch(r'```json[ \t]*\r?\n([\s\S]*?)\r?\n```',text);assert m,'Require one complete JSON or entire json fence';text=m.group(1);presentation='whole_json_fence'
 def reject(v):raise ValueError('Nonfinite JSON')
 q=json.loads(text,object_pairs_hook=unique,parse_constant=reject);assert isinstance(q,dict)
 return q,presentation
def sse(raw):
 message={'content':'','reasoning_content':''};identifier=set();usage=None;finish=None;done=False
 for line in raw.decode().splitlines():
  if not line or line.startswith(':'):continue
  assert line.startswith('data: ');text=line[6:]
  if text=='[DONE]':done=True;break
  q=json.loads(text);assert'error'not in q
  if q.get('id'):identifier.add(q['id'])
  if q.get('usage')is not None:usage=q['usage']
  for c in q.get('choices',[]):
   assert c.get('index',0)==0;delta=c.get('delta',{});assert not delta.get('tool_calls')
   for k in message:message[k]+=delta.get(k)or''
   if c.get('finish_reason')is not None:finish=c['finish_reason']
 assert done and len(identifier)==1 and finish=='stop'
 return dict(id=next(iter(identifier)),choices=[dict(message=message,finish_reason=finish)],usage=usage)
def gold(body):
 calls={};expected={};pairs=[]
 for m in body['messages']:
  if m['role']=='assistant':
   for c in m.get('tool_calls',[]):
    assert c['id']not in calls and c['function']['name']=='read_file';calls[c['id']]=json.loads(c['function']['arguments'])['path']
  if m['role']=='tool':
   name=calls[m['tool_call_id']];d=json.loads(m['content']);vals={v['id']:v['value']for v in d['records']};assert len(vals)==len(d['records'])
   selected=[vals[k]for k in d['query_ids']];assert len(selected)==4 and all(type(v)is int for v in selected)
   expected[name]=dict(nonce=d['nonce'],sum=sum(selected));pairs.append(m['tool_call_id'])
 assert set(expected)=={'left.json','right.json'}and pairs==list(reversed(calls))
 expected['total']=expected['left.json']['sum']+expected['right.json']['sum'];return expected
def percentile(xs,p):
 ys=sorted(xs);return ys[min(len(ys)-1,max(0,math.ceil(p*len(ys))-1))]if ys else None
def analyze(archive):
 files=archive['files'];verified=0
 def raw_by(suffix):
  values=[v for n,v in files.items()if n.endswith(suffix)];assert len(values)==1,(suffix,len(values));v=values[0]
  raw=v['text'].encode()if v.get('encoding','text')=='text'else __import__('base64').b64decode(v['base64']);assert len(raw)==v['bytes']and sha(raw)==v['SHA256'];return raw
 for v in files.values():
  raw=v['text'].encode()if v.get('encoding','text')=='text'else __import__('base64').b64decode(v['base64']);assert len(raw)==v['bytes']and sha(raw)==v['SHA256'];verified+=1
 read=lambda suffix:json.loads(raw_by(suffix));watch=read('-watch.json');assert watch['completed']and watch['passed']and not watch['forced_cleanup']
 assert watch['publication_commit']=='4973a6d3aed37559ff0af05c9c70048f60847ec6'and watch['deployment_source_SHA256']=='9575ff4685fb1d89195dc8e393659903c3d7a327f47ea47d516a9ba7368cb41d'
 front,ready,closed,owner=[read('/'+n)for n in ['FRONTEND.json','MODEL_READY.json','closed-engine.json','owner.json']]
 identity=front['identity'];assert all(q['identity']==identity for q in [ready,closed,owner]);assert identity['actual_step']==watch['actual_GPU']and identity['source_manifest_SHA256']=='79285a56e96c4f3586f94b3af6a56bbde085bfdfda2250c9b0335361aaaddec0'
 assert ready['Model_READY']and ready['Native_internal_owner_count']==8 and ready['HC_mode']==0
 assert owner['loaded_key_count']==67569 and owner['Native_SHA256']=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
 assert owner['completed']and owner['normal_Native_shutdown']and owner['engine_worker_shutdown_completed']and owner['observed_all8_threads_exited']
 assert closed['normal']and closed['engine_destroy_returned_zero']and closed['observed_all8_threads_exited']
 assert owner['owner_thread_observation']['status']=='OBSERVED'and len({v['TID']for v in owner['owner_thread_observation']['owners']})==8
 aff=owner['owner_affinity_after_weight_load']['actual_owner_report']['owners'];assert {v['device']for v in aff}==set(range(8))
 masks=[int(n)for v in aff for n in v['verified_mask'].split(',')];assert len(masks)==len(set(masks))==32 and all(v['changes']==1 for v in aff)
 assert set(owner['actual_providers_before_weights'].values())=={'/refactor/candidates/latest-family-native-r269/Deep/libDeep_latest_native_r269.so'}
 client=read('/result.json');assert client['completed']and client['passed']and client['attempted_HTTP']==client['completed_HTTP']==2
 fixture=read('/frozen-toolprompt.json');actual=[];bodies=[];calls=0
 for i,(thinking,budget)in enumerate([(False,4096),(True,8192)]):
  body=read('/request-%d.request.json'%i);bodies.append(body);assert body['messages']==fixture['body']['messages']and body['tools']==fixture['body']['tools']
  assert body['stream']==thinking and body['max_tokens']==budget and body['chat_template_kwargs']=={'enable_thinking':thinking}and body['tool_choice']=='none'
  wire_raw=raw_by('/response-%d.wire'%i);wire=sse(wire_raw)if thinking else json.loads(wire_raw);choice=wire['choices'][0];assert choice['finish_reason']=='stop'and not choice['message'].get('tool_calls')
  result,presentation=parsed_final(choice['message'].get('content')or'');expected=gold(body);assert result==expected
  assert type(result['total'])is int and all(type(result[k]['sum'])is int for k in ['left.json','right.json'])
  if thinking:assert choice['message'].get('reasoning_content')
  r=owner['requests'][i];c=client['rows'][i];assert wire['id']==r['request_id']==c['request_id']and sha(wire_raw)==c['wire_SHA256']
  h=r['admission']['history_render_audit'];assert h['qualified']and h['prequeue_bound']and not h['markers_sent_to_Native']and h['same_prepared_encoder_reproduces_prompt_and_IDs']and h['content_one_contribution_each_and_ordered']and h['all_content_contribution_slices_equal']
  assert h['output_budget']==r['max_new_tokens']==budget and bool(h['pending_thinking_prefix_open'])==thinking
  ids=r['generated_token_ids'];usage=wire['usage'];assert ids[-1]==1 and len(ids)==usage['completion_tokens']==r['actual_generated_tokens']<=budget
  assert r['prompt_tokens']==usage['prompt_tokens']==h['prompt_tokens']and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
  assert r['normal_reset_before']and r['normal_reset_after']and r['native_finish_reason']=='eos_token'and not r.get('error')
  assert r['actual_forward_attempts']==r['actual_forward_completed'];calls+=r['actual_forward_completed']
  pre=[v for v in r['Native_continuation']if v['kind']=='prefill'];assert pre[0]['past']==0 and pre[-1]['total']==r['prompt_tokens']and all(v['tokens']==1024 for v in pre[:-1])and sum(v['tokens']for v in pre)==r['prompt_tokens']
  visible=[t for token,t in zip(ids,r['backend_token_monotonic'])if token!=1];itls=[b-a for a,b in zip(visible,visible[1:])]
  actual.append(dict(index=i,thinking=thinking,budget=budget,prompt_tokens=r['prompt_tokens'],generated_tokens=len(ids),visible_tokens=len(visible),natural_EOS=True,task_correct=True,presentation=presentation,independent_gold_SHA256=sha(compact(expected)),actual_prompt_IDs_SHA256=h['Native_prompt_token_ids_SHA256'],Native_forward_host_summary=r['Native_forward_host_summary'],backend_TTFT_s=r['backend_token_monotonic'][0]-r['started_monotonic'],backend_visible_decode_tok_s=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 else None,backend_ITL_p95_s=percentile(itls,.95),backend_ITL_p99_s=percentile(itls,.99),client_HTTP_wall_s=c['client_clock']['wall_s'],client_first_payload_delta_s=c['client_clock']['first_payload_delta_s'],transport_text_buffered=True))
 assert len(owner['requests'])==2 and owner['HC_dispatch_checked_forwards']==owner['layout_dispatch_checked_forwards']==calls
 post=watch['post'];assert len(post)==1 and post[0]['node']=='a02r4n01'and not post[0]['kfd_pids']and not post[0]['expected_PIDs_still_exist']and not post[0]['matching_component_processes']and post[0]['ports_bindable']=={'43882':True}
 accounts=archive['accounting'];required=[watch['actual_GPU'],watch['CPU_keeper'],watch['actual_client'],post[0]['actual_step']]
 for step in required:
  rows=[r for r in accounts if r[0]==step];assert len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0']
 return dict(passed=True,scope='Ordinary public Deep HC0/afc6 TP8PP1 two complete tool histories, HTTP4096 and thinkingSSE8192; not benchmark/performance or incrementalSLO',actual_GPU=watch['actual_GPU'],CPU_keeper=watch['CPU_keeper'],actual_client=watch['actual_client'],original_GPU_epoch=watch['original_GPU_start_unix'],publication=watch['publication_commit'],verified_raw_files=verified,actual_HTTP=2,Native_CAPI_forwards=calls,all8internalowners_normal=True,post_clean=True,loading_seconds=owner['loading_seconds'],requests=actual,finite_scope='Original producer finite BF16 checks ran for every completed forward; no independent activation oracle in this public test',performance_comparison=False,fullbenchmark=False,incremental_client_latency_qualified=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();raw=a.archive.read_bytes();value=json.loads(raw);digest=sha(raw)
 if 'sealed_archive'in value:
  seal=value['sealed_archive'];original=seal['text'].encode();assert len(original)==seal['bytes']and sha(original)==seal['SHA256'];value=json.loads(original);digest=seal['SHA256']
 q=analyze(value);q['archive_SHA256']=digest;q['local_archive_wrapper_SHA256']=sha(raw)
 result=json.dumps(q,indent=2)+'\n'
 if a.output:a.output.write_text(result,encoding='utf8')
 print(result,end='')
if __name__=='__main__':main()
