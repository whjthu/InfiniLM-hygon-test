"""CPU source-contract only; no Native/OS affinity change."""
import json
from types import SimpleNamespace
from service_affinity_r146 import verify_service_affinity
class Report:
 def __init__(self,owners):self.owners=owners
 def __call__(self):return json.dumps({'owners':self.owners}).encode()
def library(bound=False):
 return SimpleNamespace(modern_owner_affinity_report_r2392=Report([{'device':i,'changes':int(bound),'verified_mask':'1,2,3,4'if bound else''}for i in range(8)]))
tests=[];strict=[]
def check(lib):strict.append(True);return {'strict_called':True}
q=verify_service_affinity(library(),{'agent_requests':[]},0,check);assert q['verified'] and not q['active_inference_affinity_qualified'] and not strict;tests.append('idle_unbound_observed_not_active_qualified')
q=verify_service_affinity(library(True),{'agent_requests':[{'real_model':True}]},1,check);assert q['strict_called'] and len(strict)==1;tests.append('active_original_strict_verifier_called')
for report,calls,lib in [({'agent_requests':[]},1,library()),({'agent_requests':[]},0,library(True)),({},0,library())]:
 try:verify_service_affinity(lib,report,calls,check)
 except AssertionError:pass
 else:raise AssertionError('idle bypass accepted contradictory evidence')
 tests.append('contradictory_or_missing_idle_evidence_rejected')
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'Native':False,'affinity_changed':False,'cases':tests}))
