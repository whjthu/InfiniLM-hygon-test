import pathlib,json,hashlib,re,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{}}
def clean(q):
 if isinstance(q,dict):return {k:clean(v)for k,v in q.items()if not any(x in k.lower()for x in ['nonce','argv','args','environment','transport_secret'])}
 if isinstance(q,list):return [clean(x)for x in q]
 if isinstance(q,str):return re.sub(r'(?i)\b[0-9a-f]{24,}\b','<hex-redacted>',q)
 return q
def read(p,kimi=False):
 if not p.exists():out['files'][str(p.relative_to(R))]={'exists':False};return
 assert p.stat().st_size<8*1024**2
 b=p.read_bytes();q=json.loads(b);out['files'][str(p.relative_to(R))]={'SHA256':hashlib.sha256(b).hexdigest(),'size':len(b),'data':clean(q)if kimi else q}
for name in ['Deep-agent-P0-r143c-watch.json','results/deep-agent-p0-r143c-910374.1145.json','results/deep-agent-p0-r143c-910374.1145-identity.json','results/GLM-context-count-only-r9392.json','GLM-agent-service-r143b-watch.json']:
 read(R/name)
for p in sorted((R/'results').glob('deep-agent-p0-r143c-910374.1145*')):
 if p.is_dir():
  for f in p.rglob('result.json'):read(f)
read(R/'results/Kimi-agent-service-r144c-watch.json',True)
for pattern in ['Kimi-agent-service-r144c-kimi-agent-p0-r144c-a91b0c6e-node*.json','Kimi-agent-service-ready-r144c-kimi-agent-p0-r144c-a91b0c6e.json','Kimi-agent-service-identity-r144c-kimi-agent-p0-r144c-a91b0c6e.json']:
 for p in sorted((R/'results').glob(pattern)):read(p,True)
print(json.dumps(out))
