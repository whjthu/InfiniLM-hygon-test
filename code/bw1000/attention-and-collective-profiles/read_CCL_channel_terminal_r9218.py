import pathlib,json,hashlib,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');p=r/'results/CCL-channel-control-r9217.json';b=p.read_bytes();c=json.loads(b);assert c['completed']and c['qualified'];out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'control_SHA256':hashlib.sha256(b).hexdigest(),'control':c,'results':[],'network_evidence':[]}
for arm,v in c['arms'].items():
 for row in v['results']:
  p=pathlib.Path(row['path']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==row['SHA256'];out['results'].append({'arm':arm,'node':row['node'],'SHA256':row['SHA256'],'data':json.loads(b)})
 for p in sorted((r/'logs').glob('CCL-channel-r9217-'+arm+'-'+v['handle']+'-*.log')):
  text=p.read_text(errors='replace');keep=[l for l in text.splitlines()if any(t in l for t in ['Using network','NCHANNELS','coll channels','CCL_RAW_AR_R9217_NORMAL','NCCL WARN','NCCL ERROR'])];out['network_evidence'].append({'arm':arm,'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'lines':keep,'used_IBext_v8':'Using network IBext_v8'in text,'Socket_data':'Using network Socket'in text})
print(json.dumps(out))
