import threading,time
from native_engine_bridge_r143c import NativeEngineBridge
class ServiceQueueBridge(NativeEngineBridge):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self._journal_lock=threading.Lock();self._journal_ready=threading.Condition(self._journal_lock);self._journal={};self._peak=0;self._peak_P1_sessions=0
 def _check_raw_budget(self,data):
  # Original public sampling builder casts int(); reject malformed wire input
  # before that conversion can silently normalize a noninteger budget.
  for obj in[data,data.get('sampling_params')or{}]:
   for name in['max_tokens','max_completion_tokens','max_new_tokens']:
    value=obj.get(name)
    if value is not None and(type(value)is not int or not 1<=value<=8192):raise ValueError('GLM output budget must be integer1..8192; no clamp')
 def _check_limit(self,params):
  value=getattr(params,"max_tokens",None);value=self.default_max_tokens if value is None else value
  if type(value)is not int or not 1<=value<=8192:raise ValueError("GLM output budget must be integer1..8192; no clamp")
 def validate_data(self,data,sampling_params):
  self._check_raw_budget(data);self._check_limit(sampling_params);return super().validate_data(data,sampling_params)
 def add_chat_request(self,**kwargs):
  self._check_raw_budget(kwargs.get("request_data")or{});self._check_limit(kwargs["sampling_params"])
  req=super().add_chat_request(**kwargs);session=str((kwargs.get('request_data')or{}).get('user','P0'))
  with self._journal_lock:
   self._journal[req.request_id]={'request_id':req.request_id,'session':session,'started':False,'finished':False,'original_reset_all32_completed':False,'native_forward_calls':0}
   live=[q for q in self._journal.values()if not q['finished']];self._peak=max(self._peak,len(live));self._peak_P1_sessions=max(self._peak_P1_sessions,len({q['session']for q in live if q['session']in('P1-HTTP','P1-SSE')}))
   self._journal_ready.notify_all()
  return req
 def take(self,timeout=1.0):
  req=super().take(timeout)
  if req is not None:
   with self._journal_ready:
    if not self._journal_ready.wait_for(lambda:req.request_id in self._journal,timeout=5.):raise RuntimeError('accept journal publication missing')
  return req
 def record_started(self,req):
  with self._journal_lock:
   row=self._journal[req.request_id];assert not row['started']and not row['finished'];assert not any(q['started']and not q['finished']for q in self._journal.values());row['started']=True
 def record_finished(self,req,calls,reset_all32):
  assert reset_all32
  with self._journal_lock:
   row=self._journal[req.request_id];assert not row['finished'];row.update(finished=True,original_reset_all32_completed=True,native_forward_calls=int(calls))
 def cancel_pending_after_global_drain(self,reason):
  # STOP control already drained all32; queued requests have never entered Native.
  with self._lock:pending=[q for q in self._requests.values()if not q.is_finished()]
  for req in pending:
   with self._journal_lock:row=self._journal[req.request_id];assert not row['started']or row['finished']
   if not row['finished']:self.record_finished(req,0,True)
   req.finish(reason)
 def journal_snapshot(self):
  with self._journal_lock:return {'backend':'original Native serial FIFO; HTTP entry concurrent, not native batching','max_accepted_unfinished':self._peak,'max_distinct_P1_sessions_unfinished':self._peak_P1_sessions,'requests':[dict(q)for q in self._journal.values()],'no_tokens_or_raw_history':True}
