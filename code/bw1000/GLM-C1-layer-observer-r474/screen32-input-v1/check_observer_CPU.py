"""CPU recording only: high64 outer handle, tail-prefill refusal, cap and33requests."""
import ctypes as C,json
from observer_trial import Observer474,schedule,phases,NAMES
pointer=(1<<60)+0x12345;calls=[];keep=[]
class Context:
 ctx=pointer;ready=True;past=0;rank=15
 def owned(self):assert self.ctx==pointer
 def check(self,rc):assert rc==0
 def error(self):return'CPU recording'
class Library:pass
lib=Library();state={'global_tp_rank':15,'frame_cap':3,'event_cap_per_frame':512,'total_frames_used':0,'pending_next_decode':False}
def mk(name):
 def fn(p,*args):
  assert p==pointer;calls.append((name,list(args)))
  if name=='next_decode':state['total_frames_used']+=1
  return 0
 return fn
for k,(name,args,ret)in NAMES.items():
 if k=='read':
  def rd(p):
   assert p==pointer;calls.append(('read',[]));keep.append(C.create_string_buffer(json.dumps(state).encode()));return C.addressof(keep[-1])
  f=C.CFUNCTYPE(C.c_void_p,*args)(rd)
 else:f=C.CFUNCTYPE(C.c_int,*args)(mk(k))
 setattr(lib,name,f);keep.append(f)
c=Context();a=Observer474(lib,c);a.prepare();a.policy(True)
for past,complete in [(1248,False),(1248,True)]:
 c.past=past
 try:a.before_AR_continuation(1249,complete)
 except AssertionError:pass
 else:raise AssertionError('partial fullprompt tail improperlymarked')
assert a.mark_calls==0
for i in range(4):c.past=1249+i;assert a.before_AR_continuation(1249,True)is(i<3)
assert a.read()['total_frames_used']==3;c.past=0;a.policy(False);a.policy(True);c.past=1500;assert a.before_AR_continuation(1249,True)is False;c.past=0;a.policy(False)
s=schedule();assert len(s)==33 and sum(a['diagnostic']for a in s)==1 and[a['case_index']for a in s if not a['diagnostic']]==list(range(32))and len(phases())==37 and all(a['owner_cores']==4 for a in s)
print(json.dumps(dict(passed=True,CPU_recording_only=True,high64_outer_handle=True,partialprefill_tail_refused=True,AR_framecap3_survives_policyreset=True,planned33=True,screen32_uniqueindices=True,barriers37=True,Native=False,GPU=False)))
