"""Narrow actual observer regression contract, not another CPU762 run."""
import json
from observation import observe,deadline_owned_active
def main():
 results=[]
 for transient in [False,True]:
  watch={'GPU_submitted':True};now=[0];items=([TimeoutError('temporary sacct loss')]if transient else [])+[['910371.1','owned','RUNNING','0:0','a02r3n03'],['910371.1','owned','COMPLETED','0:0','a02r3n03']]
  def exact():
   x=items.pop(0)
   if isinstance(x,Exception):raise x
   return x
  def sleep(n):now[0]+=n
  row=observe(exact,watch,lambda:None,20,lambda:now[0],sleep);assert row[2]=='COMPLETED'and len(watch.get('observation_errors',[]))==int(transient);results.append(dict(case='transient_then_terminal'if transient else'normal_terminal',passed=True))
 watch={'GPU_submitted':True};assert deadline_owned_active(lambda:None,watch,10,11)is None
 watch['actual_GPU']='910371.1';active=['910371.1','owned','RUNNING','0:0','a02r3n03']
 assert deadline_owned_active(lambda:active,watch,10,9)is None
 assert deadline_owned_active(lambda:active,watch,10,11)=='910371.1'
 assert deadline_owned_active(lambda:['910371.2',*active[1:]],watch,10,11)is None
 def lost():raise TimeoutError('state unavailable')
 assert deadline_owned_active(lost,watch,10,11)is None and 'deadline_state_unknown_no_cancel'in watch
 results.extend(dict(case=n,passed=True)for n in ['unknown_handle_no_cancel','before_bound_no_cancel','fresh_known_owned_after_bound_only','different_handle_no_cancel','unobservable_no_cancel'])
 print(json.dumps(dict(completed=True,passed=True,GPU=0,Slurm=False,Native_dlopen=False,CPU762_not_repeated=True,cases=results)))
if __name__=='__main__':main()
