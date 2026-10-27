import pathlib,json,hashlib,datetime,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=r/'results/Kimi-agent-token-count-r143.json';out['KimiTokenCount']={'exists':p.exists()}
if p.exists():out['KimiTokenCount'].update(receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),receipt=json.loads(p.read_bytes()))
p=r/'Deep-prefix-native-once-r136-watch.json';out['DeepDiagnostic']={'exists':p.exists()}
if p.exists():
 w=json.loads(p.read_bytes());out['DeepDiagnostic'].update({k:w.get(k)for k in ['actual_handle','Slurm','terminal','state','qualified','error','post_empty','result','result_SHA256','result_summary']})
 if w.get('result'):
  q=json.loads(pathlib.Path(w['result']).read_bytes());out['DeepDiagnostic']['summary']={k:q.get(k)for k in ['completed','normal_close','state_metadata_qualified','numerical_quality','model_quality','error','hard_failures','unrecoverable','MTP_head']}
rows=subprocess.check_output(['sacct','-S','2026-10-27','-j','910379,910374,910407','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines();out['Slurm']=[s.split('|')for s in rows if any(n in s for n in ['Kimi-agent-token-count-r143','Deep-prefix-native-once-r136','Deep-prefix-native-keeper-r136'])];print(json.dumps(out))
