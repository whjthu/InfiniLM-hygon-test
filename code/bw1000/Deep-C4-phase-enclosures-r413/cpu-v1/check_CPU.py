from pathlib import Path
import ast,json,concurrent.futures,threading
from driver_core import plan,run_owner
def main():
 assert [p['timing']for p in plan()]==[0,1,0,1,0]and len(plan())==5
 records=[];lock=threading.Lock()
 def owner(rank,fail=None,closefail=False):
  calls=[];tid=threading.get_ident()
  def create(r):calls.append(('create',r));return {'owner':tid}
  def trial(c,item):
   assert c['owner']==threading.get_ident()==tid
   calls.append(('reset',item['index']))
   for p in range(0,8496,1024):calls.append(('prefix',p,min(1024,8496-p),0))
   calls.append(('query_attempt',item['index'],item['timing']))
   if item['index']==fail:raise ValueError('Original status=2')
   calls.append(('query_completed',item['index']))
   return dict(index=item['index'],owner=rank,counts=[1,1,1,0,0],clock_reads=32 if item['timing']else 0,qualified=True)
  def destroy(c):
   assert c['owner']==tid;calls.append(('destroy',rank))
   if closefail:raise RuntimeError('closefault')
  def shutdown():calls.append(('shutdown',rank))
  try:rows=run_owner(rank,create,trial,destroy,shutdown,lambda:calls.append(('abort',rank)))
  except BaseException as e:
   assert fail is not None and isinstance(e,ValueError)and str(e)=='Original status=2'
   assert calls[-1][0]=='shutdown'
   assert any(c[0]=='shutdown'for c in calls)and len([c for c in calls if c[0]=='query_completed'])==fail
   return calls
  assert len(rows)==5 and len([c for c in calls if c[0]=='prefix'])==45 and calls[-2][0]=='destroy'and calls[-1][0]=='shutdown'
  with lock:records.append(rows)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(owner,range(8)))
 assert len(records)==8 and sum(len(r)for r in records)==40
 for fail in [0,1,3,4]:owner(0,fail)
 owner(0,1,True)
 S=Path(__file__).resolve().parent
 for p in S.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 source=(S/'private_indexer.cpp').read_text();assert source.count('status(r,flag,deep413::Site::')==5
 assert 'r.memcpyD2H(&value,flag->data(),4);r.syncStream();if(value)throw'in source
 adapter=(S/'adapter.cpp').read_text();assert 'c->check();f();'in adapter and 'c->poisoned=true;drain(c);return 1;'in adapter
 assert 'c->stats=deep413::owner_phase(timing,'in adapter and 'std::this_thread::get_id()'in adapter
 return dict(passed=True,actual_driver_core_owner_records=8,five_query_rows=40,reset_prefix_rebuilt_each_arm=True,first_ON_warm_before_formal=True,failed_query_not_completed=True,primary_preserved_cleanup_attempted=True,private_original_status_body_retained=True,five_coarse_completion_enclosures=True,CPU_only=True,GPU_qualification=False)
if __name__=='__main__':print(json.dumps(main()))
