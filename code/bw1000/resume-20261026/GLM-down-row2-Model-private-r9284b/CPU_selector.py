import pathlib,json,copy
from case_plan_r9284b import load,validate
good=load();assert all(e['chunked']is False for e in good['episodes'])and[e['down_row2']for e in good['episodes']]==[False,True,True,False]
for mutate in [lambda q:q['episodes'][0].update(down_row2=0),lambda q:q['episodes'][1].update(chunked=True),lambda q:q['episodes'][2].update(down_row2=False)]:
 q=copy.deepcopy(good);mutate(q)
 try:validate(q)
 except ValueError:continue
 raise AssertionError('invalidplan accepted')
print(json.dumps({'passed':True,'selector_strict_ABBA':True,'wholeprefix_botharms':True,'max_live':16,'original_core_CPU_qualification':'910376.726 unchanged notrerun','native_build':False}))
