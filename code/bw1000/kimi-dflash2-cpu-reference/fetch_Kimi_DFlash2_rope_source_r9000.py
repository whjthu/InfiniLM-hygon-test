import pathlib,urllib.request,hashlib,json
r=pathlib.Path.cwd()/'Kimi-DFlash2-CPU-private-r9000';u='https://raw.githubusercontent.com/lightseekorg/TorchSpec/main/torchspec/models/draft/deepseek_eagle.py'
with urllib.request.urlopen(u,timeout=20)as f:raw=f.read(180000)
assert len(raw)<180000;(r/'torchspec_deepseek_eagle.py').write_bytes(raw);print(json.dumps({'url':u,'SHA256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}))
