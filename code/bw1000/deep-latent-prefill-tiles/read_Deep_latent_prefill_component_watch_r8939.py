import pathlib,json,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-latent-prefill-component-r8939';d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sourceclaim':'02f00cb8ca30c6326b8855e26db8fcf808d2ffe7','scope':'originalwatch only, finite event/device rows+hash, noModel/weights/CCL'}
for key in ['launch','watch']:
 p=r/(label+'-'+key+'.json')
 if p.exists():
  z=json.loads(p.read_text());d[key]={k:v for k,v in z.items()if k not in ['args','CPU_result_audit_stdout','CPU_result_audit_stderr','error_warning_lines']}
  if z.get('CPU_result_audit_stdout'):
   try:d['original_audit']=json.loads(z['CPU_result_audit_stdout'])
   except ValueError:d['audit_parse_error']=True
h=d.get('watch',{}).get('actual_handle')
if h:
 for k,p in [('report',r/'results'/(label+'-'+h+'.json')),('log',r/'logs'/(label+'-'+h+'.log'))]:
  if p.exists():
   raw=p.read_bytes();d[k+'_SHA256']=hashlib.sha256(raw).hexdigest();d[k+'_bytes']=len(raw)
   if k=='report':d[k]=json.loads(raw)
   else:
    rows=[]
    for s in raw.decode(errors='replace').splitlines():
     if s.startswith('{'):
      try:rows.append(json.loads(s))
      except ValueError:pass
    d['native_rows']=rows;d['error_types']=[x for x in ['Traceback','AssertionError','RuntimeError','Segmentation fault']if x in raw.decode(errors='replace')]
print(json.dumps(d))