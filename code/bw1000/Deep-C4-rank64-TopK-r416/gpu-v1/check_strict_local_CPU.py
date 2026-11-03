from pathlib import Path
import sys,json,copy
import numpy as np
import strict_topk

def main():
 def packet(values,ids=None):
  scores=np.asarray(values,np.float32);idx=np.argsort(scores)[::-1][:512]if ids is None else np.asarray(ids,np.int64);vals=scores[idx].copy()
  def f(a):return dict(raw=np.ascontiguousarray(a).view(np.uint8).reshape(-1),shape=a.shape)
  return dict(fields={4:f(scores),5:f(idx.astype(np.int64)),11:f(vals)})
 good=packet(np.linspace(-1,1,2124,dtype=np.float32));detail=[{}];strict_topk.check([good],detail);assert detail[0]['TopK_exact_ranked_value_set_on_actual_F32_scores']
 tie=packet(np.ones(2124,np.float32),np.arange(512,dtype=np.int64));strict_topk.check([tie],[{}])
 cases=[]
 for name in ['wrong_set','wrong_order','wrong_values']:
  p=copy.deepcopy(good)
  if name=='wrong_set':
   ids=p['fields'][5]['raw'].view(np.int64);ids[-1]-=1;p['fields'][11]['raw'].view(np.float32)[-1]=p['fields'][4]['raw'].view(np.float32)[ids[-1]]
  elif name=='wrong_order':
   for key in [5,11]:p['fields'][key]['raw']=p['fields'][key]['raw'].reshape(512,-1)[::-1].copy().reshape(-1)
  else:p['fields'][11]['raw'][0]^=1
  try:strict_topk.check([p],[{}])
  except AssertionError:cases.append(name);continue
  raise AssertionError(('negative passed',name))
 return dict(passed=True,actual_strict_TopK_consumer=True,finite_ranked_set_and_sorted_order=True,tie_ID_nonexact_accepted=True,negative_cases=cases,Native_dlopen=False,GPU=False)
if __name__=='__main__':print(json.dumps(main()))
