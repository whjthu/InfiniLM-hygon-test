import hashlib,json,urllib.request
base='https://huggingface.co/tencent/HunyuanVideo-1.5/raw/efe2609403af902268e12e0e6d9a54ae73d1884a/transformer/480p_t2v/'
out={}
for n,limit in [('diffusion_pytorch_model.safetensors',1024),('config.json',8192)]:
    with urllib.request.urlopen(base+n,timeout=12) as r:raw=r.read(limit+1)
    assert len(raw)<=limit
    if n.endswith('safetensors'):
        fields=dict(s.split(' ',1) for s in raw.decode().splitlines())
        out['weight']={'bytes':int(fields['size']),'SHA256':fields['oid'].removeprefix('sha256:')}
    else:
        assert isinstance(json.loads(raw),dict)
        out['config']={'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}
print(json.dumps(out))
