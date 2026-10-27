"""Client-only contracts, never reported as model or HTTP service validation."""
import importlib.util,pathlib,json,tempfile,hashlib,datetime
D=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('client',D/'validate_tools_http_r136.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def event(delta=None,finish=None):return 'data: '+json.dumps({'choices':[{'index':0,'delta':delta or{},'finish_reason':finish}]})
lines=[event({'role':'assistant'}),event({'reasoning_content':'hidden reasoning'}),event({'tool_calls':[{'index':0,'id':'call_1','type':'function','function':{'name':'calculate','arguments':'{"a":'}}]}),event({'tool_calls':[{'index':0,'function':{'arguments':'2,"b":3}'}}]}),event(finish='tool_calls'),'data: [DONE]']
message,finish,_=m.decode_sse(lines);assert finish=='tool_calls'and message['content']==''and message['reasoning_content']=='hidden reasoning'and json.loads(message['tool_calls'][0]['function']['arguments'])=={'a':2,'b':3}
checks=['SSE_fragmented_arguments_reasoning_separation']
for label,bad in [('missing_DONE',lines[:-1]),('missing_finish',lines[:-2]+['data: [DONE]']),('error_event',['data: '+json.dumps({'error':{'message':'truncated'}}),'data: [DONE]'])]:
 try:m.decode_sse(bad)
 except AssertionError:checks.append('reject_'+label)
 else:raise AssertionError(label)
with tempfile.TemporaryDirectory(prefix='tools-client-contract-',dir=D)as tmp:
 p=pathlib.Path(tmp);(p/'input.json').write_text('{"a":2,"b":3,"nonce":"contract-only"}');audit=[]
 def call(name,args):return {'id':'contract_'+name,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}
 assert json.loads(m.execute_tool(call('read_file',{'path':'input.json'}),p,audit)['content'])['nonce']=='contract-only'
 assert json.loads(m.execute_tool(call('calculate',{'a':2,'b':3}),p,audit)['content'])['result']==5
 m.execute_tool(call('write_file',{'path':'result.json','content':'{"nonce":"contract-only","sum":5}'}),p,audit);assert json.loads((p/'result.json').read_text())['sum']==5;checks.append('sandbox_real_read_calculate_write')
 for name,args in [('read_file',{'path':'../input.json'}),('write_file',{'path':'../result.json','content':'{}'}),('calculate',{'a':True,'b':3})]:
  try:m.execute_tool(call(name,args),p,audit)
  except AssertionError:pass
  else:raise AssertionError('unsafe argument accepted')
 checks.append('reject_path_escape_and_bool_integer')
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'scope':'CLIENT_ONLY_CONTRACT_NO_HTTP_NO_MODEL_NO_GPU','checks':checks,'client_SHA256':hashlib.sha256((D/'validate_tools_http_r136.py').read_bytes()).hexdigest(),'GPU':False,'HTTP_requests':0,'model_outputs':0}
p=D/'CPU-client-contract.json';assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q))
