"""Offline stdlib audit of the original ordinary public API two-request seal.

Read-only. No collect-root option, remote polling, tokenization, or device import.
The seal contains exact original bytes; it is private R/L evidence, not Git data.
"""
import argparse,base64,gzip,hashlib,json,math

def digest(raw):return hashlib.sha256(raw).hexdigest()
def audit(seal, *, expected_commit='6d109899ecdd5add18e06c846fb724b0c46e0347',
          expected_deployment='9cdd952ae5c87c5c56ec10af43ef682c11e6e589c0b8625044088a3b0dfae914',
          expected_provider='122b69d7c1571f1523e15c672839e2d4479e91863e80018b4a8bb60c2ce93724'):
 items=seal['files'];checks=[]
 def raw(name):
  row=items[name];value=base64.b64decode(row['base64'],validate=True)
  assert digest(value)==row['SHA256']and len(value)==row['bytes'],name
  return value
 def data(name):return json.loads(raw(name))
 for name in items:raw(name)
 watch=data('watch.json');assert watch['completed']and watch['passed']and not watch['forced_cleanup']
 assert watch['publication_commit']==expected_commit
 assert watch['deployment_source_SHA256']==expected_deployment
 assert watch['provider_source_SHA256']==expected_provider
 for key in ['actual_GPU','actual_client','CPU_keeper']:assert watch[key]
 assert watch['HTTP_planned']==2 and watch['warms']==0 and not watch['reload']and not watch['retry']
 for name in ['provider','deployment']:
  manifest=data(name+'/SOURCE_MANIFEST.json')
  expected=watch['provider_source_SHA256']if name=='provider'else watch['deployment_source_SHA256']
  assert digest(raw(name+'/SOURCE_MANIFEST.json'))==expected
  for path,h in manifest['files_SHA256'].items():assert digest(raw(name+'/'+path))==h
 ready=data('run/MODEL_READY.json');identity=ready['identity']
 assert identity['actual_step']==watch['actual_GPU']and identity['source_manifest_SHA256']==watch['provider_source_SHA256']
 assert ready['actual_owner_count']==16 and ready['Model_READY']and ready['all16_loaded_postload4GiB_and_service_agreement']
 assert data('run/STOP.json')['identity']==identity and data('run/STOP.json')['normal_stop']
 assert ready['Native_SHA256']=='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
 eos=set(ready['prepared_identity']['eos_token_ids']);card=data('deployment/FIXTURE_CARD.json')
 client=data('client/result.json');assert client['completed']and client['passed']and client['attempted_HTTP']==client['completed_HTTP']==len(client['rows'])==2
 expected=[('anchor-calculate','calculate',{'a':94200,'b':70238}),('long-write','write_file',{'path':'result.json','content':{'nonce':'630b27f7f3c52777','first_sum':164438,'second_sum':84428}})]
 for index,(name,tool,gold)in enumerate(expected):
  f=card['fixtures'][index];original=raw('fixtures/'+f['local_untracked_request_file']);body=json.loads(original)
  assert digest(original)==f['request_SHA256']and body['max_tokens']==4096
  records=json.loads(next(m['content']for m in body['messages']if m['role']=='tool'and'records'in m['content']))
  values={r['id']:r['value']for r in records['records']};a,b,c,d=[values[k]for k in records['query_ids']]
  assert all(type(v)is int for v in [a,b,c,d])
  assert (gold=={'a':a,'b':b})if index==0 else(gold['content']==dict(nonce=records['nonce'],first_sum=a+b,second_sum=c+d))
  wire=raw('client/response-%d.wire'%index)
  if index==0:response=json.loads(wire);message=response['choices'][0]['message'];finish=response['choices'][0]['finish_reason'];usage=response['usage']
  else:
   calls={};finish=usage=None;done=False
   for line in wire.decode().splitlines():
    if not line or line.startswith(':'):continue
    assert line.startswith('data: ')
    if line[6:]=='[DONE]':done=True;break
    q=json.loads(line[6:]);assert'error'not in q
    if q.get('usage')is not None:usage=q['usage']
    for choice in q.get('choices',[]):
     if choice.get('finish_reason')is not None:finish=choice['finish_reason']
     for part in choice.get('delta',{}).get('tool_calls',[]):
      v=calls.setdefault(part.get('index',0),dict(id='',function=dict(name='',arguments='')))
      v['id']+=part.get('id','')
      for field in ['name','arguments']:v['function'][field]+=part.get('function',{}).get(field,'')
   assert done;message=dict(tool_calls=[calls[k]for k in sorted(calls)])
  assert finish=='tool_calls'and len(message['tool_calls'])==1
  call=message['tool_calls'][0];assert call['id']and call['function']['name']==tool
  arguments=json.loads(call['function']['arguments'])
  if index:arguments['content']=json.loads(arguments['content'])
  assert arguments==gold
  assert all(type(arguments[k])is int for k in ['a','b'])if index==0 else all(type(arguments['content'][k])is int for k in ['first_sum','second_sum'])
  history={c['id']for m in body['messages']for c in m.get('tool_calls',[])};assert call['id']not in history
  for m in body['messages']:
   if m['role']=='tool':assert m['tool_call_id']in history
  assert usage['prompt_tokens']==f['current_prepared_prompt_tokens']and 0<usage['completion_tokens']<=4096
  assert usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
  assert all(type(usage[k])is int for k in ['prompt_tokens','completion_tokens','total_tokens'])
  assert digest(wire)==client['rows'][index]['wire_SHA256']
  cc=client['rows'][index]['client_clock'];assert cc['sent']<=cc['headers']<=cc['completed']and cc['wall_s']>0
 config=data('deployment/config.json');signatures=[];native_calls=0;emitter=None;loading=[]
 for stage in range(2):
  masks_stage=[]
  for rank in range(8):
   report=data('run/owner-%d-%d.json'%(stage,rank));life=data('run/lifecycle-%d-%d.json'%(stage,rank))
   assert report['actual_step']==watch['actual_GPU']and report['source_manifest_SHA256']==watch['provider_source_SHA256']
   assert report['completed']and report['normal_shutdown']and report['worker_shutdown_completed']and report['post_close_Torch_liveness']
   assert life['completed']and life['normal']and life['destroy_attempts']==life['shutdown_attempts']==1 and life['destroy_RC']==life['shutdown_RC']==0
   assert report['all16_preweight_8GiB_passed']and report['all16_loaded_and_postload_4GiB_passed']and report['memory_after_load'][0]>4*1024**3
   assert report['memory_after_create'][0]>8*1024**3
   assert report['sdk_sha256']==config['SDK_SHA256']and report['actual_FP8_helpers']==config['FP8_helpers_SHA256']
   binding=report['actual_r403_Native_binding'];assert binding['library_SHA256']==config['Native_SHA256']and len(binding['CAPI_providers'])==8 and binding['actual_mapped_Native']
   assert set(binding['CAPI_providers'].values())=={config['Native_library']}
   assert report['local_constructor_load_finish_completed']and report['all_local_Native_keys_loaded']and report['loaded_parameters']>0
   assert report['actual_layer_indices']==list(range((0,38)[stage],(38,78)[stage]))
   assert report['constructor_noV_verified']and report['constructor_noV_census']['cache_category_bytes'].get('projected_value',0)==0
   assert report['registry_owner_before_weightIO']['all16passed']and report['registry_destroyed']
   cache=report['cache_after_destroy'];assert cache['created']==cache['destroyed']and cache['active']==cache['active_bytes']==0
   ready_rank=data('run/ready-%d-%d.json'%(stage,rank));assert ready_rank['identity']==identity and ready_rank['loaded']and ready_rank['postload_4GiB']and ready_rank['service_agreement']
   loading.append(dict(stage=stage,rank=rank,loading_s=report['loading_seconds'],parameters=report['loaded_parameters'],source_bytes=report['source_parameter_bytes']))
   def masks(value):return value['width'],sorted((r['device'],r['changes'],r['verified_mask'])for r in value['actual_owner_report']['owners'])
   first=report['owner_affinity_after_load'];final=report['owner_affinity_final'];assert masks(first)==masks(final)
   mine=[r for r in final['actual_owner_report']['owners']if r['device']==rank];assert len(mine)==1 and mine[0]['changes']==1 and len(mine[0]['verified_mask'].split(','))==4
   masks_stage.extend(map(int,mine[0]['verified_mask'].split(',')))
   rows=report['service_requests'];assert report['attempted_Native']==report['completed_Native']==len(rows)==2
   sig=[]
   for index,row in enumerate(rows):
    assert row['generated_natural_EOS']and row['reason']=='eos_token'and row['original_reset_all16_completed']
    assert row['native_profile']==('glm-nov16-w2','glm-nov16-w4')[index]and row['actual_prompt_token_ids_SHA256']==card['fixtures'][index]['current_prepared_prompt_IDs_SHA256']
    for c in row['native_calls']:
     native_calls+=1;assert c['CAPI_returned']and c['CAPI_RC']==0 and math.isfinite(c['Native_CAPI_s'])and c['Native_CAPI_s']>=0
     assert c['finite']is True if stage==1 else c['finite']is None
    sig.append((row['actual_prompt_token_ids_SHA256'],row['generated_token_ids_SHA256'],row['generated_tokens']))
   signatures.append(sig)
   if(stage,rank)==(1,0):emitter=rows
  assert len(masks_stage)==len(set(masks_stage))==32
 assert all(sig==signatures[0]for sig in signatures)
 timings=[]
 for index,row in enumerate(emitter):
  tokens=row['generated_token_ids'];ticks=row['backend_token_monotonic'];assert len(tokens)==len(ticks)==row['generated_tokens']and tokens[-1]in eos
  assert all(type(t)is int and t>=0 for t in tokens)and not any(t in eos for t in tokens[:-1])
  assert row['normal_reset_before']and row['normal_reset_after']and row['native_finish_reason']=='eos_token'
  assert row['request_id']==client['rows'][index]['request_id']
  assert client['rows'][index]['usage']['completion_tokens']==len(tokens)==row['actual_generated_tokens']
  assert row['started_monotonic']<=ticks[0]<=ticks[-1]<=row['ended_monotonic']
  assert digest(json.dumps(tokens,separators=(',',':')).encode())==row['generated_token_ids_SHA256']
  assert all(b>=a for a,b in zip(ticks,ticks[1:]))and all(math.isfinite(v)for v in ticks)
  visible=[v for t,v in zip(tokens,ticks)if t not in eos]
  timings.append(dict(fixture=expected[index][0],backend_TTFT_s=ticks[0]-row['started_monotonic'],visible_tokens=len(visible),including_EOS=len(tokens),backend_visible_decode_tps=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 else None,client_wall_s=client['rows'][index]['client_clock']['wall_s']))
 assert len(watch['post'])==2
 for row in watch['post']:assert not row['kfd_pids']and not row['matching_component_processes']and not row['expected_PIDs_still_exist']and row['ports_bindable']=={'43843':True}
 for key in ['actual_GPU','actual_client','CPU_keeper']:
  rows=[r for r in seal['accounting']['rows']if r.split('|')[0]==watch[key]];assert len(rows)==1 and rows[0].split('|')[2:4]==['COMPLETED','0:0']
 return dict(completed=True,passed=True,verified_original_files=len(items),external_HTTP_engine_requests=2,actual_owner_count=16,actual_Native_CAPI_calls=native_calls,natural_EOS_gold=True,normal16_destroy=True,independent_two_node_postclean=True,timings=timings,loading=loading,
  no_fresh_causal_speedup=True,online_SLO=False,incremental_text_streaming=False,independent_tokenizer_decode=False,
  scope='Actual ordinary public API two complete tool requests; raw wire/tool integers and Native IDs/EOS/ticks independently checked; tokenizer/parser model semantics inherited rather than reloaded')

def main():
 p=argparse.ArgumentParser();p.add_argument('seal');p.add_argument('--expected-commit');p.add_argument('--expected-deployment');p.add_argument('--expected-provider');a=p.parse_args()
 with gzip.open(a.seal,'rt',encoding='utf-8')as f:q=json.load(f)
 pins={k:v for k,v in [('expected_commit',a.expected_commit),('expected_deployment',a.expected_deployment),('expected_provider',a.expected_provider)]if v is not None}
 print(json.dumps(audit(q,**pins),indent=2,allow_nan=False))
if __name__=='__main__':main()
