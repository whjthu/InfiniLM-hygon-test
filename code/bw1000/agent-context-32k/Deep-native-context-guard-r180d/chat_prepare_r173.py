"""One official encoding/context contract for CPU fixtures and live requests."""
import copy,hashlib,json,pathlib,importlib.util
D=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('official_dsv4_r173',D/'encoding/encoding_dsv4.py');encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
def prepare(messages,data,add_generation_prompt,kwargs,tokenizer,context=None,limit=None):
 caller=messages;original=copy.deepcopy(messages);messages=copy.deepcopy(messages);tools=kwargs.get('tools')
 if tools:
  if not messages or messages[0].get('role')!='system':messages.insert(0,{'role':'system','content':''})
  messages[0]['tools']=tools
 for message in messages:
  if isinstance(message.get('content'),list):message['content']=''.join(p['text']for p in message['content'])
  for call in message.get('tool_calls')or[]:
   if isinstance(call['function'].get('arguments'),dict):call['function']['arguments']=json.dumps(call['function']['arguments'],ensure_ascii=False,allow_nan=False)
 mode=kwargs.get('thinking_mode','thinking'if kwargs.get('enable_thinking',True)else'chat')
 prompt=encoder.encode_messages(messages,thinking_mode=mode,reasoning_effort=kwargs.get('reasoning_effort'))
 if not add_generation_prompt:prompt=prompt.removesuffix('<｜Assistant｜>'+('<think>'if mode=='thinking'else'</think>'))
 ids=list(tokenizer.encode(prompt,add_special_tokens=False));assert ids and all(type(x)is int and x>=0 for x in ids)
 assert caller==original # Encoding copies; caller history is never changed.
 if context is not None:
  if type(limit)is not int or limit<1 or len(ids)+limit>context:raise ValueError('Complete prompt plus output budget exceeds actual Native context; no cropping')
 return prompt,ids
def memory(capacity,rows=128):
 assert capacity in [1024,2048]and rows==128
 paged=lambda slots,block:((slots+block-1)//block)*(((block*584+575)//576)*576)
 hca=paged(rows*((capacity+127)//128),2)*20
 csa=paged(rows*((capacity+3)//4),64)*21
 index=((rows*((capacity+3)//4)+63)//64)*8448*21
 return dict(capacity=capacity,rows=rows,SWA_window=128,HCA_layers=20,CSA_layers=21,HCA_compressed_per_row=(capacity+127)//128,CSA_compressed_per_row=(capacity+3)//4,HCA_paged_bytes_per_owner=hca,CSA_paged_bytes_per_owner=csa,indexer_paged_bytes_per_owner=index,total_capacity_dependent_bytes_per_owner=hca+csa+index,SWA_and_compressor_state='unchanged; ring window128',allocator_and_transient='not included; observe actual all8 mem_get_info')
