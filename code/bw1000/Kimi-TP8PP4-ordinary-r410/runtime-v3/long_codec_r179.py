"""Original exact K3 codec with explicit normal-agent service capabilities."""
from xtml_codec_r143 import KimiXTMLCodec
class KimiLongCodec(KimiXTMLCodec):
 def __init__(self,model_dir):
  super().__init__(model_dir)
  if self.identity['tokenizer_config_SHA256']!='5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e' or self.identity['tiktoken_model_SHA256']!='b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103':raise ValueError('actual official checkpoint tokenizer SHA mismatch')
  self.model_max_length=32768
  self.identity.update(service_context=32768,ordinary_default_output_budget=4096,thinking_output_budget=8192,thinking_default_effort='high',honored_thinking_efforts=['low','high','max'],capacity_GPU_qualified=False)
 def encode_chat(self,messages,chat_template_kwargs=None,add_generation_prompt=True):
  kwargs=dict(chat_template_kwargs or {})
  if 'thinking' in kwargs and 'enable_thinking' in kwargs and kwargs['thinking']!=kwargs['enable_thinking']:raise ValueError('conflicting thinking options')
  if 'thinking_effort' in kwargs and 'reasoning_effort' in kwargs and kwargs['thinking_effort']!=kwargs['reasoning_effort']:raise ValueError('conflicting thinking effort options')
  thinking=kwargs.get('thinking',kwargs.get('enable_thinking',True))
  if type(thinking)is not bool:raise ValueError('thinking must be boolean')
  effort=kwargs.get('thinking_effort',kwargs.get('reasoning_effort'))
  if effort is not None and effort not in ['low','high','max']:raise ValueError('Kimi supports thinking effort low/high/max only')
  if thinking and effort is None:kwargs['thinking_effort']='high'
  kwargs['preserve_history_reasoning']=True
  return super().encode_chat(messages,kwargs,add_generation_prompt)
