import pathlib,json,tempfile
from case_plan_r9260 import load
p=pathlib.Path(__file__).parent;good=load();assert [(e['label'],e['chunked'])for e in good['episodes']]==[('A0-whole',False),('B0-chunk32',True),('B1-chunk32',True),('A1-whole',False)]
print(json.dumps({'passed':True,'selector_strict_ABBA':True,'original_core_CPU_qualification':'910376.726 unchanged not rerun','native_build':False}))
