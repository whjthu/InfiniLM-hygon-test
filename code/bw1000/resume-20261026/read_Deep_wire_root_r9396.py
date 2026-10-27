import pathlib,json,hashlib,base64
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'results/deep-agent-p0-r143c-910374.1145-client';q={'files':{},'names':[str(p.relative_to(D))for p in sorted(D.rglob('*'))if p.is_file()]}
for p in sorted(D.iterdir()):
 if p.is_file()and any(x in p.name for x in ['007','002']):
  assert p.stat().st_size<256*1024;b=p.read_bytes();q['files'][str(p.relative_to(D))]={'SHA256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'base64':base64.b64encode(b).decode()}
for suffix in ['preweight','evidence']:
 p=R/'results'/('deep-agent-p0-r143c-910374.1145-'+suffix+'.json')
 if p.exists():
  b=p.read_bytes();assert len(b)<1024**2;q[suffix]={'SHA256':hashlib.sha256(b).hexdigest(),'data':json.loads(b)}
print(json.dumps(q))
