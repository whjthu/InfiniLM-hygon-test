"""Explicit32768 full-history admission. No truncation or reduced budgets."""
from native_engine_bridge_r143 import NativeEngineBridge
CONTEXT=32768

def validate_prompt_ids(ids,vocab_size):
 if not 1<=len(ids)<CONTEXT:raise ValueError('full prompt must leave output room inside32768; no truncation')
 if any(type(i)is not int or not 0<=i<vocab_size for i in ids):raise ValueError('Kimi prompt ID outside actual vocabulary')
class KimiLongBridge(NativeEngineBridge):
 def __init__(self,*args,**kwargs):
  kwargs.setdefault('default_max_tokens',4096)
  super().__init__(*args,**kwargs)
  if self.max_context!=32768 or self.default_max_tokens!=4096:raise ValueError('r179 context32768/default4096 required')
 def _limits(self,data,params):
  budget=getattr(params,'max_tokens',None);budget=self.default_max_tokens if budget is None else int(budget)
  if not 1<=budget<=8192:raise ValueError('r179 output budget1..8192, normal default4096; requested budget is never lowered')
  kwargs=data.get('chat_template_kwargs')or{};effort=kwargs.get('thinking_effort',kwargs.get('reasoning_effort',data.get('reasoning_effort')))
  if effort is not None and effort not in ['low','high','max']:raise ValueError('Kimi thinking effort unsupported; no silent fallback')
 def validate_data(self,data,sampling_params):
  self._limits(data,sampling_params);return super().validate_data(data,sampling_params)
 def add_chat_request(self,*,messages,sampling_params,request_id,request_data=None,add_generation_prompt=True,chat_template_kwargs=None):
  self._limits(request_data or{},sampling_params)
  return super().add_chat_request(messages=messages,sampling_params=sampling_params,request_id=request_id,request_data=request_data,add_generation_prompt=add_generation_prompt,chat_template_kwargs=chat_template_kwargs)
