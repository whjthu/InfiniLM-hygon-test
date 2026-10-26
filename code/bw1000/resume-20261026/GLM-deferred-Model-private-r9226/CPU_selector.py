import pathlib,json,copy,hashlib,sys
from case_plan_r9226 import load,validate
q=load();assert [x['deferred_feedback']for x in q['cohorts']]==[False,True,True,False,False,True];assert [x['C']for x in q['cohorts']]==[32]*4+[16]*2
n=0
for area in ['warmups','cohorts']:
 for i,row in enumerate(q[area]):
  for bad in [0,1,None,'1',[],{}]:
   b=copy.deepcopy(q);b[area][i]['deferred_feedback']=bad
   try:validate(b);raise AssertionError('accepted nonbool')
   except ValueError:n+=1
for bad in [0,8,32,True]:
 b=copy.deepcopy(q);b['cohorts'][0]['M']=bad
 try:validate(b);raise AssertionError('accepted invalidM')
 except ValueError:n+=1
print(json.dumps({'passed':True,'negative_cases':n,'ABBA_and_singlemicrocontrol':True,'warmorder_serial_then_deferred':True,'defaulthelper_OFF':True,'CPU_only':True}))
