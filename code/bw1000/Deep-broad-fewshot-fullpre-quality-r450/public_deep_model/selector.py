"""Explicit per-request transport selection, before Native FIFO publication."""
from .incremental_text import IncrementalTextMixin
def selection(data):
 if 'text_delivery'not in data:
  return 'P1'if data.get('stream')is True else'P0'
 value=data['text_delivery']
 if type(value)is not str or value not in ['P0','P1']:
  raise ValueError('text_delivery must be explicit P0 or P1; default P1 only for actual stream=true')
 return value
class TextDeliveryMixin:
 def validate_data(self,data,sampling_params):
  selection(data)
  return super().validate_data(data,sampling_params)
 def add_chat_request(self,**kwargs):
  mode=selection(kwargs.get('request_data')or{})
  # The existing owner acquires this same lock before taking admission metadata;
  # it cannot observe a request with a missing or changed delivery selection.
  with self._admission_lock:
   req=super().add_chat_request(**kwargs)
   req.metadata['text_delivery']=mode
   return req
 async def stream_request(self,req,timeout=100.,request_timeout=1000.):
  mode=req.metadata['text_delivery']
  if mode=='P1':consumer=IncrementalTextMixin.stream_request(self,req,timeout=timeout,request_timeout=request_timeout)
  elif mode=='P0':consumer=super().stream_request(req,timeout=timeout,request_timeout=request_timeout)
  else:raise RuntimeError('Prepared text delivery selection changed')
  try:
   async for item in consumer:yield item
  finally:
   await consumer.aclose()
