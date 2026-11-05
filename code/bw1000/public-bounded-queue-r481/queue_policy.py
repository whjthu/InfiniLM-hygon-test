"""Fixed experimental FIFO bounds. No new scheduler, Native call or timer."""
import threading,time,queue
class QueueCapacity(RuntimeError):pass
class QueuePolicyMixin:
 def configure_queue_policy(self,config):
  with self._admission_lock:
   if hasattr(self,'_queue_policy'):raise RuntimeError('Queue policy already fixed')
   if not self._pending.empty()or self._requests:raise RuntimeError('Queue policy requires unused engine')
   enabled=config.get('bounded_queue_enabled',False)
   if type(enabled)is not bool:raise ValueError('Explicit boolean queue policy required')
   cap=config.get('physical_pending_capacity',2);ttl=config.get('queue_wait_seconds',5.)
   if type(cap)is not int or not 1<=cap<=8 or type(ttl)not in[int,float]or not 0<ttl<=3600:raise ValueError('Invalid fixed capacity/queue TTL')
   import math
   if not math.isfinite(ttl):raise ValueError('Finite queue TTL required')
   self._queue_policy=dict(enabled=enabled,physical_pending_capacity=cap,queue_wait_seconds=float(ttl));self.queue_ledger=[];self._stopped_dequeued=[]
 def queue_policy_ready(self):return dict(self._queue_policy)
 def add_chat_request(self,**kwargs):
  with self._admission_lock:
   if not self.is_healthy():return super().add_chat_request(**kwargs)
   data=kwargs.get('request_data')or{}
   for obj in [data,data.get('sampling_params')or{},data.get('chat_template_kwargs')or{}]:
    if any(k in obj for k in ['bounded_queue_enabled','physical_pending_capacity','queue_wait_seconds']):raise ValueError('Queue policy is startup-fixed, not a request field')
   p=self._queue_policy;size=self._pending.qsize()
   if p['enabled']and size>=p['physical_pending_capacity']:
    self.queue_ledger.append(dict(request_id=kwargs['request_id'],outcome='capacity_rejected',physical_pending=size,at=time.monotonic()))
    raise QueueCapacity('physical_queue_capacity_exhausted; no Native request admitted')
   req=super().add_chat_request(**kwargs);now=time.monotonic()
   req.metadata.update(queue_admitted_at=now,queue_expires_at=now+p['queue_wait_seconds'],queue_state='queued',queue_policy=dict(p));self.queue_ledger.append(dict(request_id=req.request_id,outcome='accepted',at=now,physical_pending=self._pending.qsize()));return req
 def claim_queued(self,req):
  # Only original owner invokes this, under admission lock, after previous
  # request's original reset/drain has finished. Never finish active work here.
  p=self._queue_policy;now=time.monotonic();expired=p['enabled']and now>=req.metadata['queue_expires_at'];cancelled=req.cancel_event.is_set()
  if p['enabled']and(expired or cancelled):
   if expired and not cancelled:req.mark_abort(self.reasons.TIMEOUT)
   req.metadata.update(queue_state='skipped_before_Native',queue_dequeued_at=now,queue_skip_no_Native=True)
   req.finish(req.abort_reason or self.reasons.TIMEOUT)
   self.queue_ledger.append(dict(request_id=req.request_id,outcome='queued_cancel'if cancelled else'queued_expired',at=now,owner_TID=threading.get_native_id(),Native_entered=False));return False
  req.metadata.update(queue_state='active',queue_dequeued_at=now)
  self.queue_ledger.append(dict(request_id=req.request_id,outcome='owner_claimed',at=now,owner_TID=threading.get_native_id(),Native_entered='Only after original worker enters original policy.request'));return True

 def defer_stopped_dequeued(self,req):
  with self._admission_lock:self._stopped_dequeued.append(req)
 def close_never_owned_after_reset(self,error=None):
  # Original owner calls ONLY after successful original final reset/drain.
  # Error branch fails requests instead of inventing normal cleanup.
  with self._admission_lock:
   super().stop();items=self._stopped_dequeued;self._stopped_dequeued=[]
   count=self._pending.qsize()
   for _ in range(count):
    try:req=self._pending.get_nowait()
    except queue.Empty:break
    if req is not None:items.append(req)
   for req in items:
    if req.is_finished():continue
    if error is not None:req.fail(error);outcome='queue_failed_on_owner_failure'
    else:req.mark_abort(self.reasons.CANCELED);req.finish(req.abort_reason or self.reasons.CANCELED);outcome='queue_STOP_finished_after_reset'
    req.metadata.update(queue_state=outcome,queue_skip_no_Native=True)
    self.queue_ledger.append(dict(request_id=req.request_id,outcome=outcome,Native_entered=False,at=time.monotonic(),owner_TID=threading.get_native_id()))
