import pathlib,json,hashlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
p=pathlib.Path('/public/home/aclsylqidf/shared/Kimi-K3/config.json');d=json.loads(p.read_text());cfg=d.get('text_config',d)
keys=['attn_res_block_size','num_hidden_layers','hidden_size','rms_norm_eps']
print(json.dumps({'CPU_only':True,'config_path':str(p),'config_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'config':{k:cfg.get(k)for k in keys},'config_extra_attn_res':{k:v for k,v in cfg.items()if 'attn_res'in k}}))
