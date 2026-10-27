import pathlib,json,hashlib,subprocess,datetime,re
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=r/'candidates/toolcall-Qwen06-r136/CPU-preflight.json';out['ToolCPU']={'exists':p.exists()}
if p.exists():out['ToolCPU'].update(receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),receipt=json.loads(p.read_bytes()))
p=r/'Deep-prefix-native-once-r135-watch.json';out['Deep']={'exists':p.exists()}
if p.exists():
 q=json.loads(p.read_bytes());out['Deep'].update({k:q.get(k)for k in ['state','actual_handle','Slurm','terminal','qualified','error','post_empty','result_summary','result','result_SHA256']})
 if q.get('result'):
  x=json.loads(pathlib.Path(q['result']).read_bytes());out['Deep']['result_details']={k:x.get(k)for k in ['device_contract','completed','normal_close','all_cases_passed','error']};out['Deep']['case_coverage']=[{k:c.get(k)for k in ['name','passed','position','accepted','error']}for c in x.get('cases',[])]
parent=subprocess.check_output(['scontrol','show','job','910374','-o'],text=True,timeout=20);assert 'UserId=aclsylqidf('in parent
out['tool_resource_parent']={k:re.search(r'(?:^| )'+k+r'=([^ ]+)',parent).group(1)for k in ['JobId','JobState','EndTime','NodeList','NumNodes','NumCPUs']}
rows=subprocess.check_output(['sacct','-S','2026-10-27','-j','910374,910407','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines();out['Slurm']=[s.split('|')for s in rows if any(n in s for n in ['toolcall-Qwen06-CPU-r136','Deep-prefix-native-once-r135','Deep-prefix-native-keeper-r135'])];print(json.dumps(out))
