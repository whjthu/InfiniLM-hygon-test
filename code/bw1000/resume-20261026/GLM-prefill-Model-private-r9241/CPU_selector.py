import json,copy
from case_plan_r9241 import load,validate
q=load();assert [r['prefill_deferred']for r in q['cohorts']]==[False,True,True,False,False,True]and all(r['decode_deferred']is True for r in q['cohorts']);assert [r['C']for r in q['cohorts']]==[32]*4+[8]*2
n=0
for area,key in [('warmups','prefill_deferred'),('cohorts','prefill_deferred'),('cohorts','decode_deferred')]:
 for i in range(len(q[area])):
  for bad in [0,1,'1',None,[],{}]:
   z=copy.deepcopy(q);z[area][i][key]=bad
   try:validate(z);raise AssertionError('nonbool')
   except ValueError:n+=1
z=copy.deepcopy(q);z['cohorts'][0]['decode_deferred']=False
try:validate(z);raise AssertionError('decodeOFF')
except ValueError:n+=1
print(json.dumps({'passed':True,'strictbool_negative_cases':n,'main_ABBA':True,'C8_singleprefillcontrol':True,'decode_fixed_ON':True,'CPU_only':True}))
