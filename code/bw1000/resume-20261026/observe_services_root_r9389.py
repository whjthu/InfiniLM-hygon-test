import pathlib,json,subprocess,datetime,hashlib
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'watches':{},'CPU_receipts':{},'account':{}}
keys=['UTC','state','actual_handle','CPU_actual_step','CPU_keeper_step','Slurm','GPU_submitted','completed','post_empty','normal_service_lifecycle','Native_lifecycle_qualified','HTTP_tool_qualified','run','watch_ready_before_GPU','preweight_passed','service_ready','terminal']
for name in ['GLM-agent-service-r143b-watch.json','results/Kimi-agent-service-r144c-watch.json','Deep-agent-P0-r143c-watch.json']:
 p=R/name
 if p.exists():
  o=json.loads(p.read_bytes());z={k:o[k]for k in keys if k in o};z['error_present']=bool(o.get('error'));z.pop('service_ready',None);z['service_ready_present']=bool(o.get('service_ready'));q['watches'][name]=z
 else:q['watches'][name]={'exists':False}
p=R/'candidates/Deep-agent-p0-r143c/CPU-preflight.json'
if p.exists():q['CPU_receipts']['Deep']={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'receipt':json.loads(p.read_bytes())}
for parent,prefix in [('910374','Deep-agent-P0'),('910376','GLM-agent-service'),('910380','Kimi-agent-service'),('910407','agent')]:
 a=subprocess.run(['sacct','-S','2026-10-27','-j',parent,'-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed,NodeList%110'],capture_output=True,text=True,timeout=30)
 q['account'][parent]={'exit':a.returncode,'rows':[x for x in a.stdout.splitlines()if prefix.lower()in x.lower()and ('r143b'in x or 'r143c'in x or 'r144c'in x)]}
print(json.dumps(q))
