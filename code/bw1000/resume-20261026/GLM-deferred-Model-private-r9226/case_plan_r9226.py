import pathlib,json

def validate(q):
 if type(q) is not dict or set(q)!={'schema','warmups','cohorts'}or q['schema']!=1:raise ValueError('plan schema')
 expected=[('serialwarm',False),('deferredwarm',True)]
 if len(q['warmups'])!=2 or len(q['cohorts'])!=6:raise ValueError('bounded plan')
 for row,(label,flag) in zip(q['warmups'],expected):
  if set(row)!={'label','deferred_feedback'}or row['label']!=label or type(row['deferred_feedback'])is not bool or row['deferred_feedback']is not flag:raise ValueError('strict warm bool/order')
 expected=[('C32M16-A0',32,False),('C32M16-B0',32,True),('C32M16-B1',32,True),('C32M16-A1',32,False),('C16M16-control-A',16,False),('C16M16-control-B',16,True)]
 for row,(label,C,flag)in zip(q['cohorts'],expected):
  if set(row)!={'label','C','M','deferred_feedback'}or row['label']!=label or type(row['C'])is not int or row['C']!=C or type(row['M'])is not int or row['M']!=16 or type(row['deferred_feedback'])is not bool or row['deferred_feedback']is not flag:raise ValueError('strict cohort selector/order')
 return q

def load():return validate(json.loads(pathlib.Path(__file__).with_name('case_plan_r9226.json').read_bytes()))
