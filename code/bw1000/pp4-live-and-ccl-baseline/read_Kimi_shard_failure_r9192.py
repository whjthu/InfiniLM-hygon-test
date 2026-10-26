import pathlib,json,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run='shard-r9192-3dfd61b66fd538b3';out={'observedUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'nodes':[]}
p=r/'results'/('Kimi-shard-model-r9192-'+run+'.json');w=json.loads(p.read_bytes());out['health']={k:v for k,v in w.items()if k not in ['groups','draft']};out['groups']=[{k:v for k,v in g.items()if k!='args'}for g in w.get('groups',[])];out['draft']={k:v for k,v in w.get('draft',{}).items()if k!='args'}
for n in range(4):
 p=r/'results'/('Kimi-shard-PP2-r9192-'+run+'-node%d.json'%n);raw=p.read_bytes();w=json.loads(raw);out['nodes'].append({'rawSHA256':hashlib.sha256(raw).hexdigest(),**{k:w[k]for k in ['node','phase','error','traceback','completed','head_after_finish','memory_after_finish','memory_after_create','loading_seconds','loaded_keys','all8_owned_workers_shutdown_completed','shutdown_error','serving_protocol_complete','owner_affinity','batch_cases','head_after_protocol']if k in w}})
print(json.dumps(out,ensure_ascii=False))
