import pathlib,json,hashlib,subprocess,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name,rel,keys in [('GLM','GLM-critical-Model-r133-watch.json',['state','actual_handle','Slurm','terminal','qualified','error','audit_qualified','post_passed']),('Deep','Deep-prefix-native-once-r135-watch.json',['state','actual_handle','Slurm','terminal','qualified','error','post_empty','result_summary']),('KimiPool','results/Kimi-prefill-pool-r133-control.json',['CPU_actual_step','actual_handle','terminal','error','results','post_KFD','error_post_KFD'])]:
 p=r/rel;v={'exists':p.exists()}
 if p.exists():
  q=json.loads(p.read_bytes());v.update({k:q.get(k)for k in keys});v['evidence_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
  if name=='GLM':v['warmups']={k:{n:x.get(n)for n in ['stage','phase','completed','UTC_epoch','stall_guard_active']}for k,x in q.get('warmups',{}).items()}
 out[name]=v
for label,folder in [('TP32j','Kimi-TP32-construct-r120j'),('Deep135CPU','Deep-prefix-device-scope-r135')]:
 p=r/'candidates'/folder/'CPU-build.json';out[label]={'exists':p.exists()}
 if p.exists():out[label].update(receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),receipt=json.loads(p.read_bytes()))
rows=subprocess.check_output(['sacct','-S','2026-10-27','-j','910374,910376,910379,910407','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines();out['Slurm']=[s.split('|')for s in rows if any(n in s for n in ['GLM-critical-Model-r133','GLM-critical-Model-keeper-r133','Kimi-prefill-pool-r133','Deep-prefix-native-once-r135','Deep-prefix-native-keeper-r135','Kimi-TP32-constructor-direct-link-r120j'])];print(json.dumps(out))
