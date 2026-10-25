import pathlib,json,struct,hashlib
r=pathlib.Path.home()/'shared/Kimi-K3';d={'path':str(r),'read_values':False}
config=r/'config.json';d['config_SHA256']=hashlib.sha256(config.read_bytes()).hexdigest();cfg=json.loads(config.read_text());d['config']={k:cfg.get(k)for k in ['hidden_size','vocab_size','tie_word_embeddings','torch_dtype','text_config']}
idx=r/'model.safetensors.index.json';j=json.loads(idx.read_text());d['index_SHA256']=hashlib.sha256(idx.read_bytes()).hexdigest();found={k:v for k,v in j['weight_map'].items()if k.endswith(('embed_tokens.weight','lm_head.weight'))};d['tensors']={}
for key,shard in found.items():
 p=r/shard
 with p.open('rb')as f:n=int.from_bytes(f.read(8),'little');assert n<16<<20;raw=f.read(n);h=json.loads(raw)
 d['tensors'][key]={'shard':str(p),'header_SHA256':hashlib.sha256(raw).hexdigest(),'header_bytes':n,'payload_offset':8+n,**h[key]}
base=pathlib.Path.home()/'hygon-inference-refactor-20261014';q=base/'results/DFlash2-CDN-detached-r9018.json';d['checkpoint_receipt']=json.loads(q.read_text())if q.exists()else None;print(json.dumps(d))
