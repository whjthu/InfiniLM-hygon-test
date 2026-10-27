import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');p=r/'results/deep-c4-projection-model-r5920-build.json';b=p.read_bytes();q=json.loads(b)
assert q['completed'];units=[u for u in q['units']if pathlib.Path(u['object']).name in ['unit-7.o','unit-8.o','unit-9.o']];assert len(units)==3
print(json.dumps({'read_only':True,'receipt_SHA256':hashlib.sha256(b).hexdigest(),'units':[{k:v for k,v in u.items()if k in ['source','object','source_sha256','object_sha256','depfile','dependencies']}for u in units]}))
