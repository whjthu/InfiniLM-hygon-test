import pathlib,json,hashlib,subprocess,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run='kimi-sparse-current-r136-8a216fc3';p=R/'results'/('Kimi-sparse-profile-model-r136-'+run+'.json');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exists':p.exists()}
if p.exists():
 b=p.read_bytes();w=json.loads(b);out.update(evidence_SHA256=hashlib.sha256(b).hexdigest(),completed=w.get('completed'),ready=w.get('ready'),passed=w.get('passed'),CPU_actual_step=w.get('CPU_actual_step'),error=w.get('error'),post_KFD_passed=w.get('post_KFD_passed'),manifest_SHA256=w.get('manifest_SHA256'),groups=[{k:v.get(k)for k in ['actual_handle','actual_row','terminal']}for v in w.get('groups',[])],draft={k:w.get('draft',{}).get(k)for k in ['actual_handle','actual_row','terminal','status','ready_seen']},CPU_closure=w.get('actual_CPU_closure',{}).get('terminal'))
print(json.dumps(out))
