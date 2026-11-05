"""Opt-in SSE comments; standard data frames and owned finish remain unchanged."""
import asyncio,contextvars,json,threading,time
from contextlib import suppress
HTTP_SCOPE=contextvars.ContextVar('token_clock_http_scope',default=None)
SESSION=contextvars.ContextVar('native_token_clock',default=None)
PREFIX=': infinilm-token-clock '
class Session:
 def __init__(self,request_id):
  self.id=request_id
  if HTTP_SCOPE.get()is not None:HTTP_SCOPE.get()['request_id']=request_id
  self.loop=asyncio.get_running_loop();self.queue=asyncio.Queue(maxsize=2*8192+128);self.rows=[];self.overflow=False;self.sample_ns=None;self.text_rows=[];self.observer_emit_host_ns=0;self.req=None;self.last_return=None;self.forward_returns=[];self.started=time.monotonic_ns()
 def attach(self,req,special=()):
  if self.req is not None:raise RuntimeError('One request per clock scope')
  self.req=req;self.admitted_ns=time.monotonic_ns();req._token_clock=self;original=req.emit_token;special=frozenset(special or ())
  def emit(token,visible=None):
   original(token,visible);begin=time.monotonic_ns();stamp=time.monotonic_ns();seq=len(self.rows)
   row=dict(native_forward_ordinal=len(self.forward_returns)-1,sampled_ns=self.sample_ns,request_id=self.id,ordinal=seq,token_id=token,eos=token in req._eos_ids,special=token in special,visible=(token not in req._eos_ids)if visible is None else bool(visible),native_return_ns=self.last_return,native_emit_ns=stamp)
   if seq>=req.max_new_tokens:self.overflow=True;return
   self.rows.append(row)
   try:self.loop.call_soon_threadsafe(self.notify,('token',row))
   except RuntimeError:self.overflow=True # Closed HTTP loop must not poison Native owner.
   self.observer_emit_host_ns+=time.monotonic_ns()-begin
  req.emit_token=emit
 def notify(self,item):
  try:self.queue.put_nowait(item)
  except asyncio.QueueFull:self.overflow=True
 def sampled(self):self.sample_ns=time.monotonic_ns()
 def text_progress(self,count,empty,unstable,final):
  row=dict(request_id=self.id,text_progress=True,visible_count=count,empty_delta=empty,unstable_UTF8_tail=unstable,final=final);self.text_rows.append(row);self.notify(('text',row))
 def returned(self,kind=None,past=None,count=None):
  self.last_return=time.monotonic_ns()
  if len(self.forward_returns)<32768:self.forward_returns.append(dict(ordinal=len(self.forward_returns),kind=kind,past=past,count=count,returned_ns=self.last_return))
  else:self.overflow=True
class AdmissionMixin:
 def add_chat_request(self,**kwargs):
  with self._admission_lock:
   req=super().add_chat_request(**kwargs);s=SESSION.get()
   if s is not None:s.attach(req,getattr(self.tokenizer,'all_special_ids',()))
   return req
class ASGIClock:
 def __init__(self,app,sink):self.app,self.sink=app,sink
 async def __call__(self,scope,receive,send):
  enabled=scope['type']=='http'and dict(scope.get('headers',[])).get(b'x-infinilm-token-clock')==b'1'
  if not enabled:return await self.app(scope,receive,send)
  async def observed(message):
   begin=time.monotonic_ns();await send(message);returned=time.monotonic_ns()
   if message['type']=='http.response.body'and self.sink is not None:
    import hashlib
    b=message.get('body',b'');self.sink(dict(request_id=(HTTP_SCOPE.get()or{}).get('request_id'),kind='ASGI_send',body_bytes=len(b),body_SHA256=hashlib.sha256(b).hexdigest(),attempt_ns=begin,returned_ns=returned,more_body=message.get('more_body',False),scope='ASGI transport acceptance, not physical socket flush or client receipt'))
  token=HTTP_SCOPE.set({})
  try:return await self.app(scope,receive,observed)
  finally:HTTP_SCOPE.reset(token)
class ClockServerMixin:
 def _create_app(self):
  app=super()._create_app()
  return ASGIClock(app,getattr(self,'token_clock_sink',None))if getattr(self,'token_clock_enabled',False)else app
 async def _stream_chat(self,request_id,data,http_request):
  enabled=getattr(self,'token_clock_enabled',False)and http_request.headers.get('x-infinilm-token-clock')=='1'
  if not enabled:
   async for chunk in super()._stream_chat(request_id,data,http_request):yield chunk
   return
  s=Session(request_id)
  async def original():
   context=SESSION.set(s);cancelled=False
   try:
    async for chunk in super(ClockServerMixin,self)._stream_chat(request_id,data,http_request):await s.queue.put(('wire',chunk))
   except asyncio.CancelledError:
    cancelled=True;raise
   finally:
    SESSION.reset(context)
    if not cancelled:await s.queue.put(('end',None))
  task=asyncio.create_task(original());done_frame=None;sent=0
  try:
   while True:
    kind,value=await s.queue.get()
    if kind=='end':break
    if kind=='wire':
     if value.strip()=='data: [DONE]':done_frame=value
     else:yield value
    else:
     value=dict(value,server_yield_ns=time.monotonic_ns())
     if kind=='token':sent+=1
     yield PREFIX+json.dumps(value,separators=(',',':'))+'\n\n'
   await task
   req=s.req
   yield PREFIX+json.dumps(dict(request_id=request_id,terminal=True,emitted=len(s.rows),notices_sent=sent,observer_overflow=s.overflow,owned_finished=req.is_finished()if req else False,finish_reason=getattr(req.finish_reason,'value',str(req.finish_reason))if req else None))+'\n\n'
   if done_frame is not None:yield done_frame
  finally:
   if not task.done():
    task.cancel()
    with suppress(asyncio.CancelledError):await task
   # Record only in the private sink, never a public response body or UI.
   sink=getattr(self,'token_clock_sink',None)
   if sink is not None:sink(dict(request_id=request_id,rows=s.rows,Native_forward_returns=s.forward_returns,observer_emit_host_ns=s.observer_emit_host_ns,observer_overhead_scope='emit wrapper only; excludes ASGI yield/parser/sink; no perf qualification',text_progress=s.text_rows,observer_overflow=s.overflow,server_stream_start_ns=s.started,admitted_ns=getattr(s,'admitted_ns',None),owned_finished=s.req.is_finished()if s.req else False,closed_ns=time.monotonic_ns()))
