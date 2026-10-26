import pathlib,json
def validate(q):
 if type(q)is not dict or set(q)!={'schema','warmups','cohorts'}or type(q['schema'])is not int or q['schema']!=1:raise ValueError('schema')
 if len(q['warmups'])!=2 or len(q['cohorts'])!=6:raise ValueError('boundedplan')
 for row,(label,flag)in zip(q['warmups'],[('serialprefillwarm',False),('deferredprefillwarm',True)]):
  if set(row)!={'label','prefill_deferred'}or row['label']!=label or type(row['prefill_deferred'])is not bool or row['prefill_deferred']is not flag:raise ValueError('warm strictbool/order')
 expected=[('C32-PFA0',32,False),('C32-PFB0',32,True),('C32-PFB1',32,True),('C32-PFA1',32,False),('C8-PFcontrolA',8,False),('C8-PFcontrolB',8,True)]
 for row,(label,C,flag)in zip(q['cohorts'],expected):
  if set(row)!={'label','C','M','decode_deferred','prefill_deferred'}or row['label']!=label or type(row['C'])is not int or row['C']!=C or type(row['M'])is not int or row['M']!=16 or type(row['decode_deferred'])is not bool or row['decode_deferred']is not True or type(row['prefill_deferred'])is not bool or row['prefill_deferred']is not flag:raise ValueError('cohort strictbool/geometry/order')
 return q
def load():return validate(json.loads(pathlib.Path(__file__).with_name('case_plan_r9241.json').read_bytes()))
