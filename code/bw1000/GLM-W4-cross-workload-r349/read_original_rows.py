import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';tag='GLM-regular-service-r204-b34db8d1eb674914';p=r/'results'/('glm-agent-service-r204-'+tag+'-stage3-rank0.json');raw=p.read_bytes();q=json.loads(raw);out={'owner_path':str(p),'SHA256':hashlib.sha256(raw).hexdigest(),'queue_journal':q.get('queue_journal'),'loading_seconds':q.get('loading_seconds'),'prepared_entry_identity':q.get('prepared_entry_identity'),'actual_producer':q.get('actual_producer'),'rows':[]}
for row in q.get('service_requests',[]):
 out['rows'].append({**{k:v for k,v in row.items() if k!='native_calls'},'prefill_CAPI_s':sum(x['Native_CAPI_s']for x in row.get('native_calls',[])if x['prefill']),'decode_CAPI_s':sum(x['Native_CAPI_s']for x in row.get('native_calls',[])if not x['prefill'])})
print(json.dumps(out))
