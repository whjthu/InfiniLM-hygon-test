"""CPU shape/budget/queue contracts; never model acceptance."""
import ast,enum,json,pathlib,types,sys,hashlib,tempfile
from longcontext_wire_r190b import *
from service_queue_bridge_r190b import ServiceQueueBridge
class Reasons(enum.Enum):
 EOS_TOKEN='eos_token';LENGTH='length';CANCELED='canceled';TIMEOUT='timeout';ERROR='error'
class Tok:
 def apply_chat_template(self,messages,**kw):return messages[0]['content']
 def encode(self,text,**kw):return [1]*int(text)
def mustfail(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Expected preenqueue rejection')
for limit,length in[(4096,8192),(4096,16384),(8192,8192),(8192,24576)]:
 ids=list(range(length));assert decode_wire(encode_wire(1,BEGIN,ids,limit),1,154880)==(BEGIN,ids,limit)
 chunks=prefill_chunks(ids);assert sum((x for _,x in chunks),[])==ids and all(len(x)<=512 for _,x in chunks)
 assert [i for i,_ in chunks]==list(range(0,length,512))
mustfail(lambda:encode_wire(1,BEGIN,[1]*24577,8192));mustfail(lambda:encode_wire(1,BEGIN,[1],8193))
bridge=ServiceQueueBridge(Tok(),32768,Reasons,default_max_tokens=4096)
for limit,length in[(4096,8192),(8192,16384)]:
 data={'messages':[{'role':'user','content':str(length)}]};params=types.SimpleNamespace(max_tokens=limit,temperature=0,stop=None)
 bridge.validate_data(data,params);req=bridge.add_chat_request(messages=data['messages'],sampling_params=params,request_id=str(limit),request_data=data);assert req.max_new_tokens==limit and len(req.prompt_token_ids)==length
for limit in[8193,0,1.5,True]:
 params=types.SimpleNamespace(max_tokens=limit,temperature=0,stop=None);data={'messages':[{'role':'user','content':'8'}]}
 mustfail(lambda:bridge.validate_data(data,params));mustfail(lambda:bridge.add_chat_request(messages=data['messages'],sampling_params=params,request_id='bad',request_data=data))
# Emulate the original public builder having already cast8192.9 to8192.
for field in['max_tokens','max_completion_tokens','max_new_tokens']:
 for raw in[8192.9,True,'8192',8193]:
  for nested in[False,True]:
   data={'messages':[{'role':'user','content':'8'}]}
   if nested:data['sampling_params']={field:raw}
   else:data[field]=raw
   params=types.SimpleNamespace(max_tokens=8192,temperature=0,stop=None)
   mustfail(lambda:bridge.validate_data(data,params));mustfail(lambda:bridge.add_chat_request(messages=data['messages'],sampling_params=params,request_id='rawbad',request_data=data))
d=pathlib.Path(__file__).parent;s=(d/'glm_agent_service_probe_r190b.py').read_text();assert 'stage==1'not in s and 'stage==PP-1'in s and 'range(limit)'in s and 'prefill_chunks(prompt)'in s
assert 'min(limit'not in s and '[:512]'not in s and "getattr(lib,'modern_glm_model_'+name)(ctx,0)"in s
assert 'GLM_REGULAR_SERVICE_R190B_NORMAL'in(d/'model_entry_r190b.py').read_text()and 'GLM_REGULAR_SERVICE_R190B_NORMAL'in(d/'controller_r190b.py').read_text()
assert 'GLM_REGULAR_SERVICE_R180_NORMAL'not in(d/'model_entry_r190b.py').read_text()
import importlib.util
spec=importlib.util.spec_from_file_location('wire_contract',d.parent/'GLM-regular-fixture-private-r178a/CPU_wire_contract_r178a.py')if False else None
print(json.dumps({'passed':True,'CPU_only':True,'Native':False,'GPU':False,'context':32768,'output4096_8192_unchanged':True,'no_clamp_or_crop':True,'PP4_emitter_last_stage':True,'continuous512prefill':True,'capacity_and_output_rejected_pre_enqueue':True,'formal_agent_PASS':False}))
