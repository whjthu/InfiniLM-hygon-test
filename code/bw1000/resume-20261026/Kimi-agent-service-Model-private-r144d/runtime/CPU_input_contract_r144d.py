"""CPU admission bounds only. No HTTP, Native, tokenizer, or model run."""
from pathlib import Path
from types import SimpleNamespace
import json, queue, sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kimi_input_contract_r144d import KimiNativeEngineBridge, validate_prompt_ids
reasons=SimpleNamespace(ERROR='error')
sp=lambda cap:SimpleNamespace(max_tokens=cap,temperature=0.0,stop=None)
def engine(prompt):
 def encode(messages,data,generation_prompt,kwargs):
  validate_prompt_ids(prompt,163840)
  return 'CPU fake prompt, not model qualification',prompt
 return KimiNativeEngineBridge(None,1024,reasons,encode_chat=encode,default_max_tokens=128)
tests=[]
for cap in [0,-1,513,1024]:
 for boundary in ['pre_SSE','enqueue']:
  e=engine([1])
  try:
   if boundary=='pre_SSE':e.validate_data({'messages':[]},sp(cap))
   else:e.add_chat_request(messages=[],sampling_params=sp(cap),request_id='invalid')
  except ValueError:pass
  else:raise AssertionError((cap,boundary,'accepted'))
  try:e.take(timeout=0)
  except queue.Empty:pass
  else:raise AssertionError('Invalid request was enqueued')
  tests.append({'cap':cap,'boundary':boundary,'rejected_before_queue':True})
for prompt in [[],[True],[163840],[-1],[1]*513]:
 for boundary in ['pre_SSE','enqueue']:
  e=engine(prompt)
  try:
   if boundary=='pre_SSE':e.validate_data({'messages':[]},sp(1))
   else:e.add_chat_request(messages=[],sampling_params=sp(1),request_id='invalid')
  except ValueError:pass
  else:raise AssertionError(('badprompt',boundary,'accepted'))
  try:e.take(timeout=0)
  except queue.Empty:pass
  else:raise AssertionError('Invalid prompt was enqueued')
  tests.append({'invalid_prompt':repr(prompt[:2]),'prompt_len':len(prompt),'boundary':boundary,'rejected_before_queue':True})
e=engine([1]*512);e.validate_data({'messages':[]},sp(512));r=e.add_chat_request(messages=[],sampling_params=sp(512),request_id='edge512');assert e.take(timeout=0) is r and r.max_new_tokens==512 and len(r.prompt_token_ids)==512
tests.append({'prompt':512,'cap':512,'total':1024,'accepted_both_gates':True})
print(json.dumps({'passed':True,'CPU_only':True,'HTTP':False,'Native':False,'model_qualified':False,'cases':tests}))
