"""Four functionality calls, control only at eight-owner quiescent barriers."""
import threading,time,traceback
PHASES=('defaultOFF','epilogueON_fullpreOFF','fullpreON','clear_restoreOFF')
def run_owners(factory,control,observe):
 ready=threading.Barrier(9,timeout=40);begin=threading.Barrier(9,timeout=40);end=threading.Barrier(9,timeout=40)
 owners=[None]*8;calls=[[None]*8 for _ in PHASES];closes=[None]*8;errors=[];lock=threading.Lock();started=time.perf_counter()
 def fail(where,rank,e):
  with lock:errors.append(dict(where=where,rank=rank,error=repr(e),traceback=traceback.format_exc()))
  for b in (ready,begin,end):b.abort()
 def worker(rank):
  try:
   owner=owners[rank]=factory(rank);owner.setup();ready.wait()
   for index,name in enumerate(PHASES):
    begin.wait();calls[index][rank]=owner.call(name);end.wait()
  except BaseException as e:fail('owner',rank,e)
  finally:
   try:closes[rank]=owners[rank].close()if owners[rank]is not None else dict(normal_close=True,context_created=False)
   except BaseException as e:fail('close',rank,e)
 threads=[threading.Thread(target=worker,args=(rank,),name='HC378-owner'+str(rank))for rank in range(8)]
 for t in threads:t.start()
 records=[]
 try:
  ready.wait()
  for index,name in enumerate(PHASES):
   before=control(index,name);begin.wait();end.wait();records.append(observe(index,name,before,calls[index]))
 except BaseException as e:fail('main',None,e)
 finally:
  for t in threads:t.join(timeout=45)
 joined=not any(t.is_alive()for t in threads)
 return dict(passed=not errors and joined and len(records)==4 and all(c and c.get('normal_close')for c in closes),errors=errors,records=records,closes=closes,threads_joined=joined,whole_functionality_wall_s=time.perf_counter()-started)
