"""Recompute all six original tool answers and preserve the actual quality failure."""
from pathlib import Path
import json,hashlib,argparse
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def differences(expected,actual,path=''):
 if isinstance(expected,dict)and isinstance(actual,dict):
  out=[]
  for key in sorted(set(expected)|set(actual)):
   p=path+'/'+key
   if key not in expected:out.append(dict(path=p,expected='<ABSENT>',actual=actual[key],kind='extra_field'))
   elif key not in actual:out.append(dict(path=p,expected=expected[key],actual='<ABSENT>',kind='missing_field'))
   else:out.extend(differences(expected[key],actual[key],p))
  return out
 if expected==actual:return[]
 kind='nonce'if path.endswith('/nonce')else'integer_result'if path.endswith('/sum')or path=='/total'else'value_or_structure'
 return[dict(path=path,expected=expected,actual=actual,expected_type=type(expected).__name__,actual_type=type(actual).__name__,kind=kind)]
def analyze(a):
 a=a.get('archive',a);assert a['step']=='910374.1320';files=a['raw_files'];paths=a['paths']
 for path,item in files.items():assert hashlib.sha256(item['text'].encode()).hexdigest()==item['SHA256'],path
 load=lambda role:json.loads(files[paths[role]]['text']);f=load('fixture');r=load('report');client=load('client');legacy=load('legacy');watch=load('watch')
 calls={v['id']:json.loads(v['function']['arguments'])['path']for m in f['body']['messages']for v in m.get('tool_calls',[])};gold={}
 for m in f['body']['messages']:
  if m['role']=='tool':
   data=json.loads(m['content']);values={v['id']:v['value']for v in data['records']};gold[calls[m['tool_call_id']]]=dict(nonce=data['nonce'],sum=sum(values[k]for k in data['query_ids']))
 gold['total']=sum(v['sum']for v in gold.values());assert gold==f['expected_answer']
 rows=[];allids=[]
 for i,(native,wirepaths)in enumerate(zip(r['requests'],paths['wires'])):
  request=json.loads(files[wirepaths['request']]['text']);response=json.loads(files[wirepaths['response']]['text']);message=response['choices'][0]['message'];text=message['content'].strip()
  if text.startswith('```json\n')and text.endswith('\n```'):text=text[8:-4]
  actual=json.loads(text);diff=differences(gold,actual);ids=native['generated_token_ids'];allids.append(ids)
  deltas=[[b-a for a,b in zip(old,new)]for old,new in zip(native['HC_counters_before_request'],native['HC_counters_after_request'])]
  rows.append(dict(index=i,observer=i in [3,4],Native_request_id=native['request_id'],response_id=response['id'],HTTP_user=request['user'],request_path=wirepaths['request'],request_SHA256=files[wirepaths['request']]['SHA256'],wire_path=wirepaths['response'],wire_SHA256=files[wirepaths['response']]['SHA256'],request_messages_tools_equal_fixture=request['messages']==f['body']['messages']and request['tools']==f['body']['tools'],messages_tools_SHA256=digest(dict(messages=request['messages'],tools=request['tools'])),Native_input_ID_SHA256=native['admission']['history_render_audit']['Native_prompt_token_ids_SHA256'],input_tokens=native['prompt_tokens'],max_output=request['max_tokens'],normal_reset_before=native['normal_reset_before'],normal_reset_after=native['normal_reset_after'],HC_mode=native['HC_mode'],HC_before=native['HC_counters_before_request'],HC_after=native['HC_counters_after_request'],HC_delta_all8=deltas,HC_forward_count=len(native['HC_forward_coverage']),generated_including_EOS=len(ids),output_ID_SHA256=digest(ids),natural_EOS=native['native_finish_reason']=='eos_token'and ids[-1]==1,tool_calls_returned=message.get('tool_calls'),answer=actual,gold=gold,task_correct=not diff,differences=diff,client_success_row_present=any(v['index']==i for v in client['requests'])))
 assert len(rows)==6 and [v['task_correct']for v in rows]==[True]*5+[False]
 assert len({v['Native_input_ID_SHA256']for v in rows})==len({v['messages_tools_SHA256']for v in rows})==1
 first=allids[2];last=allids[5];div=next((i for i,(x,y)in enumerate(zip(first,last))if x!=y),None)
 return dict(state='ACTUAL_GOLD_FAIL_5_OF6_DO_NOT_REPLAY',GPU='910374.1320 FAILED1/22m11s',CPU_keeper='910407.785 COMPLETED0/22m23s',original_client_error=client.get('error'),actual_HTTP_attempts=client['actual_HTTP_attempts'],actual_HTTP_completed=client['actual_HTTP_completed'],actual_Native_completed=client['actual_Native_completed'],client_gold_success_rows=len(client['requests']),rows=rows,formalOFF2_vs5_first_different_output_index_zero_based=div,formalOFF2_vs5_common_prefix_length=div if div is not None else min(len(first),len(last)),sample_prefixes={str(n):dict(all4formal_equal=len({digest(allids[i][:n])for i in [2,3,4,5]})==1,all6_equal=len({digest(ids[:n])for ids in allids})==1,prefix_ID_SHA256=[digest(ids[:n])for ids in allids])for n in [8,87]},lifetime=dict(report_normal_Native_shutdown=r.get('normal_Native_shutdown'),legacy_engine_worker_shutdown_completed=legacy.get('engine_worker_shutdown_completed'),legacy_shutdown_entered=legacy.get('shutdown_entered'),normal_server_shutdown=r['normal_server_shutdown'],cleanup=r['cleanup'],controller_post=watch.get('post'),independent_post=a['independent_post'],normal_success_only_poststamp_not_written=True,whole_lifecycle_PASS=False),causality='Allarms HC1/current976f/sameinput andsixnormalreset; no HC_OFFmath control. Cannot attribute failure toHC orobserver orhardware. Raw4window metadata may be analyzed with explicitqualityboundary, not fullcorrect-requestprofile qualification.',raw_payloads_in_Git=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output');a=p.parse_args();q=analyze(json.loads(Path(a.archive).read_bytes()));text=json.dumps(q,ensure_ascii=False,indent=2)+'\n'
 if a.output:Path(a.output).write_text(text,encoding='utf-8',newline='\n')
 else:print(text)
if __name__=='__main__':main()
