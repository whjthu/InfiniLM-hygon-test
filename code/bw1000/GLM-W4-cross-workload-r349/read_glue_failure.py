import pathlib,json
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';q={}
for step in ['910407.674','910407.676']:
 tag='GLM349-CPU-'+step;rows={}
 for kind,path in [('supervisor',r/'results'/('GLM-latest-full-owner-r290-supervisor-'+tag+'.json')),('client',r/'results'/('GLM-latest-full-owner-r290-client-'+tag)/'paired/result.json'),('owner',r/'results'/('GLM-latest-full-owner-r290-'+tag+'-stage3-rank0.json'))]:
  if path.exists():
   d=json.loads(path.read_bytes());keep=['completed','passed','error','service_error','service_traceback','attempted_HTTP','completed_HTTP','actual_HTTP_requests','automatic_CPU_client_exit_code','automatic_CPU_client_group_forced_stop','attempted_Native','completed_Native','full_requests_completed']
   rows[kind]={k:d[k]for k in keep if k in d};rows[kind]['case_count']=len(d.get('cases',[]));rows[kind]['row_count']=len(d.get('service_requests',[]));rows[kind]['phases']={p:{'exit_code':v['exit_code'],'error':v.get('result',{}).get('error')}for p,v in d.get('phases',{}).items()}
 q[step]=rows
print(json.dumps(q))
