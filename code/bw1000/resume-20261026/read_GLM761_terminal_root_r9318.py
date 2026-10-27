import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={}
for key,rel,pin in [('result','results/GLM-M15-offset8-r127-910376.761.json','5c1a28e7c00019ae1bdf802633ebf2a374fa248356596dba04745acf4f206293'),('watch','GLM-M15-offset8-r127-watch.json',None)]:
 p=r/rel;b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 if pin:assert h==pin
 out[key]={'path':str(p),'SHA256':h,'data':json.loads(b)}
assert out['watch']['data']['qualified'] and out['watch']['data']['post_passed'] and out['result']['data']['all_offsets_qualified']
print(json.dumps(out))
