"""Fake-tokenizer copy/API semantics through existing qualified CPU bridge."""
import argparse,copy,enum,hashlib,importlib.util,json,pathlib,sys,types
from api_request_constraints_r203 import request_constraints,copied_messages,make_encode_chat
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
SOURCE_MF='6f1dc599102ea993f028999a345fe18c7b654569a38f3578eecc1f3be895424c'
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def source_pin(source):
 assert sha(source/'source-manifest-r192.json')==SOURCE_MF;m=json.loads((source/'source-manifest-r192.json').read_bytes())
 for p,h in m['source_SHA256'].items():assert sha(source/p)==h
 for p,h in m['overlay_SHA256'].items():assert sha(source/'overlay'/p)==h
 return m
def run(source):
 source=pathlib.Path(source);source_pin(source);load('native_engine_bridge_r143c',source/'native_engine_bridge_r143c.py');queue=load('source_service_queue_r203',source/'service_queue_bridge_r192.py')
 class Fake:
  def __init__(self):self.seen=[]
  def apply_chat_template(self,messages,**kwargs):self.seen.append((copy.deepcopy(messages),copy.deepcopy(kwargs)));return json.dumps([messages,kwargs],ensure_ascii=False,separators=(',',':'))
  def encode(self,text,add_special_tokens=False):assert add_special_tokens is False;return list(text.encode())
 class Reasons(enum.Enum):EOS_TOKEN='eos_token';STOP_STRING='stop_string';LENGTH='length';CANCELED='canceled';TIMEOUT='timeout';ERROR='error'
 tools=[{'type':'function','function':{'name':name,'parameters':{'type':'object'}}}for name in ['fixture_a','fixture_b']]
 messages=[{'role':'system','content':'Original system.'},{'role':'user','content':'Original history.'},{'role':'assistant','reasoning_content':'Existing reasoning retained.','content':'','tool_calls':[{'id':'fixture-id','type':'function','function':{'name':'fixture_a','arguments':{'value':7}}}]},{'role':'tool','tool_call_id':'fixture-id','name':'fixture_a','content':'Original complete tool result.'}]
 kwargs={'tools':tools,'clear_thinking':False,'reasoning_effort':'high'};before=copy.deepcopy(messages);kwbefore=copy.deepcopy(kwargs);checked=0
 for choice,parallel in [('fixture_b',False),('required',True),('auto',False),('auto',True),('none',False)]:
  k=copy.deepcopy(kwargs)
  if choice=='none':k['tools']=None
  data={'messages':messages,'_agent_choice':choice,'tool_choice':{'type':'function','function':{'name':choice}}if choice=='fixture_b'else choice,'parallel_tool_calls':parallel,'chat_template_kwargs':k,'max_tokens':4096}
  t=Fake();bridge=queue.ServiceQueueBridge(t,32768,Reasons,encode_chat=make_encode_chat(t),eos_token_ids=(1,),default_max_tokens=4096,max_pending=8);params=types.SimpleNamespace(temperature=0.0,stop=None,max_tokens=4096)
  bridge.validate_data(data,params);req=bridge.add_chat_request(messages=messages,sampling_params=params,request_id='fixture-'+str(checked),request_data=data,chat_template_kwargs=k)
  copied= t.seen[-1][0];assert copied==copied_messages(messages,data,k)and t.seen[-1][1]['tools']==k['tools']and t.seen[-1][1]['clear_thinking']is False
  assert messages==before and kwargs==kwbefore and req.max_new_tokens==4096
  if choice=='fixture_b':assert len(t.seen[-1][1]['tools'])==2 and 'fixture_b'in req.prompt and 'at most one real tool call'in req.prompt
  if choice=='auto'and parallel:assert req.prompt==Fake().apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**k)
  if choice=='none':assert 'Do not invoke any tool.'in req.prompt
  if choice=='required':assert 'at least one real tool call'in req.prompt
  assert copied[-3:]==messages[-3:];req.finish(Reasons.EOS_TOKEN);bridge.stop();checked+=1
 assert not any(x in sys.modules for x in ['torch','_infinilm','infinilm'])
 return {'passed':True,'scope':'Fake-tokenizer CPU bridge/API/history-copy contract only; not actual tokenizer or Model qualification','cases':checked,'existing_GLM_full_tool_set_preserved':True,'history_and_reasoning_unchanged':True,'auto_parallel_true_byte_identical':True,'strict_parser_source_unchanged':True,'Torch_imported':False,'Native_loaded':False}
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--source',required=True);x=a.parse_args();print(json.dumps(run(pathlib.Path(x.source))))
