import pathlib,urllib.request,hashlib,json
r=pathlib.Path.cwd()/'Kimi-DFlash2-CPU-private-r9000';u='https://raw.githubusercontent.com/lightseekorg/TorchSpec/main/torchspec/models/draft/dspark.py'
with urllib.request.urlopen(u,timeout=20)as f:raw=f.read(160000)
assert len(raw)<160000;(r/'torchspec_dspark.py').write_bytes(raw);out={'url':u,'SHA256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)};(r/'MLA-primary-source.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))
