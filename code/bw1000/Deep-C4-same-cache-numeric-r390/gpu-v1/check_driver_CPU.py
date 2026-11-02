"""Actual threaded score/commit coordinator + original ctypes consumer; fake C ABI leaves only."""
from pathlib import Path
import importlib.util,threading,concurrent.futures,json,contextlib,io
D=Path(__file__).resolve().parent;S=D.parent/'cpu-v1'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
driver=load('numeric390',D/'numeric_driver.py')
with contextlib.redirect_stdout(io.StringIO()):leaves=load('interface_fixture390',S/'check_interface_CPU.py')
assert driver.plan()==[8575,8576]
checks=[];previous=None
for fail in [None,'precheck','score1']:
 previous=None;total_commits=0
 for past in driver.plan():
  barrier=threading.Barrier(8);shared=[None]*8;lock=threading.Lock();events=[];checked=[False]
  records=leaves.fixtures.fixtures(past,previous)
  def precheck(packets):
   assert len(packets)==8 and all(p is not None for p in packets)
   assert all(e[0]!='commit'for e in events)
   assert leaves.consumer.validate_same_cache(packets,leaves.oracle,previous,committed=False)['passed']
   if fail=='precheck':raise ValueError('Injected numerical consumer refusal')
   checked[0]=True
  def owner(rank):
   ctx={'record':records[rank],'stage':'before'}
   token,packet=leaves.api.begin(leaves.f,leaves.check,ctx,past,None,None,None,None);packet['rank']=rank
   def score(mode):
    if fail=='score1'and rank==3 and mode==1:raise ValueError('Injected original score failure')
    leaves.api.score(leaves.f,leaves.check,ctx,token,mode,packet)
    with lock:events.append(('score',rank,mode))
   def commit():
    assert checked[0],'Actual coordinator committed before same-cache numerical approval'
    leaves.api.commit(leaves.f,leaves.check,ctx,token,packet)
    with lock:events.append(('commit',rank))
   driver.score_then_commit(rank,packet,score,commit,barrier,shared,precheck)
  with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
   futures=[pool.submit(owner,rank)for rank in range(8)];errors=[]
   for future in futures:
    try:future.result(timeout=10)
    except BaseException as exc:errors.append(type(exc).__name__)
  commits=[e for e in events if e[0]=='commit'];total_commits+=len(commits)
  if fail:
   assert len(errors)==8 and not commits,(fail,errors,events)
   checks.append({'case':fail,'past':past,'commits':0,'all8_stopped':True});break
  assert not errors and len(commits)==8
  final=leaves.consumer.validate_same_cache(shared,leaves.oracle,previous)
  assert final['passed']and final['nextstep_committed_cache_and_state_consumed']==(previous is not None)
  checks.append({'case':'actual8thread_coordinator_interface_consumer','past':past,'commits':8,'samecache_and_nextstate':True});previous=shared
 if fail is None:assert total_commits==16
print(json.dumps({'passed':True,'checks':checks,'actual_driver_coordinator':True,'actual_ctypes_interface_and_numeric_consumer':True,'fake_CABI_leaves_only':True,'Native_dlopen':False,'GPU':False,'new_compile':False}))
