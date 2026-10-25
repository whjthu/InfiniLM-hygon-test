import pathlib,json,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-unique-SWA-r8983';out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Finiteoriginalwatch/reportmetricsandSHAonly;noactivation/logit/weight/rawtraceexport'}
for k in ['launch','watch']:
 p=r/(label+'-'+k+'.json')
 if p.exists():out[k]={n:v for n,v in json.loads(p.read_bytes()).items()if n not in ['CPU_result_audit_stdout','CPU_result_audit_stderr','error_warning_lines','fresh_before_KFD','args']}
h=out.get('watch',{}).get('actual_handle')
if h:
 for suffix,base in [('.json','results'),('.log','logs')]:
  p=r/base/(label+'-'+h+suffix)
  if p.exists():
   out[base+'_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
   if base=='results':out['report']=json.loads(p.read_bytes())
print(json.dumps(out))
