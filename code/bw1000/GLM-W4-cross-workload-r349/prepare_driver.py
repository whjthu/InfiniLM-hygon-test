import pathlib
D=pathlib.Path(__file__).parent;O=D/'owner';P=D.parent/'public-native-bridge/GLM-prefill-W2-W4-owner-r307'
s=(P/'client_driver_r290.py').read_text()
for a,b in [
 ("PHASES=[('paired','client_paired_r290.py',6)]", "PHASES=[('paired','client_crossworkload_r349.py',18)]"),
 ("'planned_HTTP':6,'planned_Native':6", "'planned_HTTP':18,'planned_Native':18,'attempted_HTTP':0,'completed_HTTP':0,'actual_Native':0"),
 ("assert n==6", "assert n==18"),
 ("q['phases'][phase]={'exit_code'", "q['attempted_HTTP']+=phase_result.get('attempted_HTTP',0);q['completed_HTTP']+=phase_result.get('completed_HTTP',0);q['phases'][phase]={'exit_code'"),
 ("n=sum(v['result']['actual_HTTP_requests']for v in q['phases'].values());", "n=sum(v['result']['actual_HTTP_requests']for v in q['phases'].values());assert q['attempted_HTTP']==q['completed_HTTP']==18;"),
 ]:
 assert a in s,a;s=s.replace(a,b)
(O/'client_driver_r290.py').write_bytes(s.encode())
# Obsolete six-request client remains as original evidence, never selected.
for p in O.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
print('AST closure PASS; source not yet frozen')
