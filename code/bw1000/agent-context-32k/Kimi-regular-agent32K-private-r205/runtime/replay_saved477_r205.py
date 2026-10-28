"""CPU-only saved59token observation. Never print IDs/text/args/nonces/errors.

Loads the actual private pure protocol/parser modules through package namespaces
without executing infinilm.__init__ (which imports model/GPU dependencies).
This is not a substitute parser and does not repair or accept failed output.
"""
import argparse,hashlib,importlib,importlib.util,json,pathlib,sys,types
D=pathlib.Path(__file__).resolve().parent.parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=pathlib.Path(a.output)
 result={'kind':'saved477_request4_strict_parser_observation','CPU_only':True,'GPU':False,'Native_imported':False,'weights_read':False,'setup_completed':False,'strict_parser_passed':False,'call_count':None,'call_count_observed_before_error':0,'whitelist_tool_names':[],'EOS':None,'error_class':None,'error_SHA256':None,'not_Model_qualification':True}
 def error(e):result.update(error_class=type(e).__name__,error_SHA256=hashlib.sha256(str(e).encode()).hexdigest())
 try:
  pins=json.loads((D/'runtime/replay_saved477_pins-r205.json').read_bytes())
  for path,h in pins['source_SHA256'].items():assert sha(D/path)==h,'replay source identity mismatch'
  assert sha(pins['node_report'])==pins['node_report_SHA256']and sha(pins['saved_request'])==pins['saved_request_SHA256']
  report=json.loads(pathlib.Path(pins['node_report']).read_bytes());body=json.loads(pathlib.Path(pins['saved_request']).read_bytes())
  assert report['node']==2 and report['TP']==16 and report['PP']==2 and report['run']==pins['run']and report['library_sha256']==pins['old_Native_SHA256']
  assert report['manifest_SHA256']==pins['old_model_manifest_SHA256']
  item=report['agent_requests'][3];assert item['client_session']=='P1-HTTP'and item['input_tokens']==8506 and item['output_budget']==4096
  assert len(body['messages'])==4 and len(body['tools'])==3 and body['max_tokens']==4096 and body['parallel_tool_calls']is False and not body['stream']
  ids=item['generated_ids'];assert len(ids)==59 and all(type(x)is int and 0<=x<163840 for x in ids)
  result['EOS']=ids[-1]in {163585,163586};assert result['EOS']
  assert item['long_prefill_progress']['input_tokens_committed']==8506
  for name,folder in [('infinilm',D/'overlay/python/infinilm'),('infinilm.server',D/'overlay/python/infinilm/server')]:
   assert name not in sys.modules
   module=types.ModuleType(name);module.__path__=[str(folder)];sys.modules[name]=module
  protocol=importlib.import_module('infinilm.server.agent_protocol')
  parser_module=importlib.import_module('infinilm.server.tool_parsers')
  assert pathlib.Path(protocol.__file__).resolve()==D/'overlay/python/infinilm/server/agent_protocol.py'
  assert pathlib.Path(parser_module.__file__).resolve()==D/'overlay/python/infinilm/server/tool_parsers/__init__.py'
  sys.path.insert(0,str(D/'runtime'))
  from long_codec_r179 import KimiLongCodec
  from long_input_r179 import validate_prompt_ids
  codec=KimiLongCodec('/shared/Kimi-K3')
  prepared,schemas,choice=protocol.prepare_agent_request(body,'kimi_k3')
  prompt,prompt_ids=codec.encode_chat(prepared['messages'],prepared.get('chat_template_kwargs')or{},bool(prepared.get('add_generation_prompt',True)))
  validate_prompt_ids(prompt_ids,163840);assert len(prompt_ids)==8506 and len(prompt_ids)+prepared['max_tokens']<=32768
  visible=[x for x in ids if x not in {163585,163586}]
  text=codec.decode(visible).strip()  # actual buffered non-SSE _chat path
  parser=parser_module.ToolCallParser(schemas,thinking=prompt.rstrip().endswith(('<think>','<|open|>think<|sep|>')),choice=choice,parallel=prepared.get('parallel_tool_calls',True),parser='kimi_k3')
  result.update(setup_completed=True,full_prompt_length_matches=True,source_and_original_evidence_pins_verified=True)
  try:
   parser.feed(text,final=True);result['strict_parser_passed']=True;result['call_count']=len(parser.calls)
  except BaseException as e:error(e)
  calls=parser.calls;result['call_count_observed_before_error']=len(calls)
  allowed={'read_file','calculate','write_file'}
  result['whitelist_tool_names']=[c['function']['name']for c in calls if c['function']['name']in allowed]
  assert 'torch'not in sys.modules and 'transformers'not in sys.modules
 except BaseException as e:error(e)
 # Only the enumerated compact result is serialized. Private request/result
 # values stay in local RAM and are never attached to exceptions or output.
 try:
  with out.open('x',encoding='utf-8')as f:f.write(json.dumps(result,indent=2)+'\n')
 except BaseException as e:error(e);result['output_written']=False
 print(json.dumps(result));return 0 if result['setup_completed']and result.get('output_written',True)else 1
if __name__=='__main__':raise SystemExit(main())
