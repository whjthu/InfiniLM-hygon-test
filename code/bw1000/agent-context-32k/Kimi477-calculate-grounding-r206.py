"""One CPU-only check of saved actual calculate args. No private value output."""
import argparse,hashlib,importlib,json,pathlib,sys,types
D=pathlib.Path('/refactor/incoming/Kimi-agent-service-Model-r205')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 result={'CPU_only':True,'GPU':False,'Native_imported':False,'weights_read':False,'setup_ok':False,'EOS':False,'strict_one_calculate':False,'a_matches_query0':False,'b_matches_query1':False,'ordered_args_match':False,'sum_matches_query01':False,'QA_passed_for_this_saved_calculate_only':False,'error_class':None,'error_SHA256':None,'no_tool_execution_or_prompt_change':True,'not_full_agent_qualification':True}
 try:
  assert sha(D/'service-source-manifest-r205.json')=='eb70d7f32eb713b1ef2187d256cb9ac8bc3166e64fb46268242d6e6d91c2caf5'
  prior=pathlib.Path('/refactor/results/Kimi477-saved59-strict-replay-r205.json')
  assert sha(prior)=='f2d9e9667376703d045d45e5c30ba7250c7881d489adfe2cae7439863350d549'
  previous=json.loads(prior.read_bytes());assert previous['setup_completed']and previous['full_prompt_length_matches']and previous['EOS']and previous['strict_parser_passed']and previous['call_count']==1 and previous['whitelist_tool_names']==['calculate']
  pins=json.loads((D/'runtime/replay_saved477_pins-r205.json').read_bytes())
  for path,h in pins['source_SHA256'].items():assert sha(D/path)==h
  assert sha(pins['node_report'])==pins['node_report_SHA256']and sha(pins['saved_request'])==pins['saved_request_SHA256']
  report=json.loads(pathlib.Path(pins['node_report']).read_bytes());body=json.loads(pathlib.Path(pins['saved_request']).read_bytes())
  assert report['node']==2 and report['run']==pins['run']and report['library_sha256']==pins['old_Native_SHA256']and report['manifest_SHA256']==pins['old_model_manifest_SHA256']
  item=report['agent_requests'][3];ids=item['generated_ids']
  assert item['client_session']=='P1-HTTP'and item['input_tokens']==8506 and item['output_budget']==4096 and len(ids)==59 and ids[-1]in {163585,163586}
  result['EOS']=True
  assert len(body['messages'])==4 and body['max_tokens']==4096 and body['parallel_tool_calls']is False and not body['stream']
  for name,folder in [('infinilm',D/'overlay/python/infinilm'),('infinilm.server',D/'overlay/python/infinilm/server')]:
   assert name not in sys.modules
   module=types.ModuleType(name);module.__path__=[str(folder)];sys.modules[name]=module
  protocol=importlib.import_module('infinilm.server.agent_protocol');parser_module=importlib.import_module('infinilm.server.tool_parsers')
  sys.path.insert(0,str(D/'runtime'))
  from long_codec_r179 import KimiLongCodec
  from long_input_r179 import validate_prompt_ids
  codec=KimiLongCodec('/shared/Kimi-K3');prepared,schemas,choice=protocol.prepare_agent_request(body,'kimi_k3')
  prompt,prompt_ids=codec.encode_chat(prepared['messages'],prepared.get('chat_template_kwargs')or{},bool(prepared.get('add_generation_prompt',True)))
  validate_prompt_ids(prompt_ids,163840);assert len(prompt_ids)==8506
  parser=parser_module.ToolCallParser(schemas,thinking=prompt.rstrip().endswith(('<think>','<|open|>think<|sep|>')),choice=choice,parallel=prepared.get('parallel_tool_calls',True),parser='kimi_k3')
  parser.feed(codec.decode([x for x in ids if x not in {163585,163586}]).strip(),final=True)
  assert len(parser.calls)==1 and parser.calls[0]['function']['name']=='calculate'
  result['strict_one_calculate']=True
  messages=body['messages'];assert messages[-1]['role']=='tool'and messages[-2]['role']=='assistant'
  original_read=messages[-2]['tool_calls'];assert len(original_read)==1 and original_read[0]['function']['name']=='read_file'and json.loads(original_read[0]['function']['arguments'])=={'path':'input.json'}
  assert messages[-1]['tool_call_id']==original_read[0]['id']
  actual=json.loads(messages[-1]['content']);queries=actual['query_ids'];records=actual['records']
  assert len(queries)>=2 and len({x['id']for x in records})==len(records)
  values={x['id']:x['value']for x in records};expected_a,expected_b=values[queries[0]],values[queries[1]]
  args=json.loads(parser.calls[0]['function']['arguments']);assert set(args)=={'a','b'}and all(type(x)is int for x in [expected_a,expected_b,args['a'],args['b']])
  result.update(setup_ok=True,a_matches_query0=args['a']==expected_a,b_matches_query1=args['b']==expected_b,ordered_args_match=args=={'a':expected_a,'b':expected_b},sum_matches_query01=args['a']+args['b']==expected_a+expected_b)
  result['QA_passed_for_this_saved_calculate_only']=result['ordered_args_match']and result['sum_matches_query01']
  assert 'torch'not in sys.modules and 'transformers'not in sys.modules
 except BaseException as e:result.update(error_class=type(e).__name__,error_SHA256=hashlib.sha256(str(e).encode()).hexdigest())
 try:
  with pathlib.Path(a.output).open('x',encoding='utf-8')as f:f.write(json.dumps(result,indent=2)+'\n')
 except BaseException as e:result.update(setup_ok=False,error_class=type(e).__name__,error_SHA256=hashlib.sha256(str(e).encode()).hexdigest())
 print(json.dumps(result));return 0 if result['setup_ok']else 1
if __name__=='__main__':raise SystemExit(main())
