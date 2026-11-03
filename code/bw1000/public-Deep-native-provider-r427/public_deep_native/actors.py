"""One same-process engine-owner thread; Native creates its eight real workers."""
import os,threading,time
from .protocol import Run
from .worker import owner_main

class Actors:
 def __init__(self,config,prepared):
  self.config=dict(config);self.prepared=prepared;self.run=Run(config);self.owner_thread=None
  self.stop_intent=threading.Event();self.failed=threading.Event();self.finished=threading.Event();self.failure=None
  self._started=self._claimed=self._stopping=False
 def _trip(self,error):
  if not self._claimed:return
  self.failure=error;self.failed.set()
  try:self.run.fail(error,stage=0)
  except BaseException as persistence:
   if hasattr(error,'add_note'):error.add_note('Deep failure record: '+repr(persistence))
  engine=self.prepared['engine']
  with engine._lock:engine._fatal=error;requests=list(engine._requests.values())
  for request in requests:
   if not request.is_finished():request.fail(error)
 def _owner(self):
  try:owner_main(self.config,self.prepared,self.stop_intent)
  except BaseException as error:self._trip(error)
  finally:self.finished.set()
 def start(self):
  if self._started:raise RuntimeError('Deep actor already started; no replay')
  self.run.write('ENGINE_OWNER.json',dict(PID=os.getpid(),Native_internal_owner_count=8,NativeRequest_serialized=False))
  self._claimed=self._started=True
  self.owner_thread=threading.Thread(target=self._owner,name='Deep-public-Native-engine-owner',daemon=False);self.owner_thread.start()
 def wait_ready(self):
  while not (self.run.path/'ready-engine.json').exists():
   if self.failed.is_set():raise self.failure
   if self.finished.is_set():raise RuntimeError('Deep owner exited before readiness')
   self.run.deadline_check('Deep readiness',minimum=120);time.sleep(.05)
  row=self.run.read('ready-engine.json');assert row['loaded']and row['loaded_key_count']==67569 and row['Native_SHA256']==self.config['Native_SHA256']and(row['HC_mode'],row['rows'],row['capacity'])==(0,1,32768)
  owners=row['affinity']['actual_owner_report']['owners'];assert len(owners)==8 and{r['device']for r in owners}==set(range(8));masks=[]
  for owner in owners:
   assert owner['changes']==1;mask=[int(x)for x in owner['verified_mask'].split(',')];assert len(mask)==4;masks+=mask
  assert len(set(masks))==32 and all(f>=16*1024**3 for f,t in row['postload_free'])
  if self.failed.is_set():raise self.failure
  return row
 def stop(self):
  if not self._started or self._stopping:return
  if self.owner_thread is threading.current_thread():raise RuntimeError('Native owner cannot join itself or HTTP')
  self._stopping=True;engine=self.prepared['engine'];Reason=self.prepared['FinishReason']
  with engine._admission_lock:
   engine.stop();self.stop_intent.set()
   with engine._lock:requests=list(engine._requests.values())
   for request in requests:
    if not request.is_finished():request.mark_abort(Reason.CANCELED)
  self.run.stop();end=time.monotonic()+min(600,max(0,self.run.remaining()))
  while self.owner_thread.is_alive()and time.monotonic()<end:self.owner_thread.join(timeout=.1)
  if self.owner_thread.is_alive():
   self._trip(TimeoutError('Deep Native owner still alive at original cleanup deadline'))
   # This explicit provider owns this public process. Native threads are never
   # unsafely cancelled; the outer controller receives a real failed process.
   os._exit(2)
  if self.failed.is_set():raise self.failure
  closed=self.run.read('closed-engine.json');assert closed['normal']and closed['engine_destroy_returned_zero']
