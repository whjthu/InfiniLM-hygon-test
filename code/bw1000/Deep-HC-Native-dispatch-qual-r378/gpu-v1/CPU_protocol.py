"""Actual protocol entry: four controls and failure cleanup; no GPU imports."""
from protocol import run_owners,PHASES
def case(fail=None):
 closed=[];seen=[]
 class Owner:
  def __init__(self,r):self.rank=r
  def setup(self):
   if fail=='setup'and self.rank==3:raise ValueError('setup negative')
  def call(self,n):
   if fail=='call'and self.rank==5 and n==PHASES[1]:raise ValueError('call negative')
   return (self.rank,n)
  def close(self):closed.append(self.rank);return dict(normal_close=True)
 def control(i,n):seen.append(n);return i
 def observe(i,n,b,v):assert b==i and v==[(r,n)for r in range(8)];return dict(name=n)
 q=run_owners(Owner,control,observe);assert q['threads_joined']and sorted(closed)==list(range(8));assert q['passed']==(fail is None)
 if fail is None:assert seen==list(PHASES)and len(q['records'])==4
 else:assert q['errors']
 return dict(case=fail or'four_phase_complete',passed=True,all8_closed=True,threads_joined=True)
if __name__=='__main__':
 import json
 print(json.dumps(dict(passed=True,GPU=False,Model=False,cases=[case(),case('setup'),case('call')])))
