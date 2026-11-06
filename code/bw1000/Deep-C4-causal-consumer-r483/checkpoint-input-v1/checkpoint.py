"""Bounded stdlib checkpoint inspection and typed loading. No library/device import."""
from pathlib import Path
import hashlib,json,struct,math,ctypes as C,time
H=lambda b:hashlib.sha256(b).hexdigest()
SIZES={'I8':1,'BF16':2,'F32':4}

def config(meta):
 old=meta['config_original'];new=meta['config_runtime']
 assert old['max_position_embeddings']==1048576 and new['max_position_embeddings']==32768
 assert {k for k in set(old)|set(new)if old.get(k)!=new.get(k)}=={'max_position_embeddings'}
 assert old['rope_scaling']==new['rope_scaling'] and old['compress_rope_theta']==new['compress_rope_theta']
 assert new['compress_ratios'][meta['layer']]==4 and meta['TP']==1 and meta['local_heads']==64
 return new

def validate_parameters(actual,meta):
 expected=meta['weights'];assert len(actual)==len(expected)==7
 assert len({p['name']for p in actual})==len(actual),'duplicate Native parameter key'
 assert {p['name']for p in actual}==set(expected),'Native required key set differs'
 for p in actual:
  e=expected[p['name']]
  assert p['shape']==e['shape']and p['dtype']==meta['dtype_codes'][e['dtype']],'Native actual shape/dtype differs'
  assert p['bytes']==e['bytes']==math.prod(e['shape'])*SIZES[e['dtype']],'parameter bytes differ'
  assert p['contiguous']and not p['loaded'],'contiguous unloaded checkpoint parameter required'

def headers(meta,root):
 root=Path(root);assert H((root/'config.json').read_bytes())==meta['original_config_SHA256']
 raw=(root/'model.safetensors.index.json').read_bytes();assert H(raw)==meta['original_index_SHA256'];index=json.loads(raw)['weight_map'];result={}
 for name,pin in meta['headers'].items():
  assert Path(name).name==name and name.endswith('.safetensors')
  p=root/name;assert p.stat().st_size==pin['file_bytes']
  with p.open('rb')as f:
   n=struct.unpack('<Q',f.read(8))[0];assert n==pin['header_bytes']and n<=16777216;b=f.read(n)
  assert len(b)==n and H(b)==pin['header_SHA256'];result[name]=(json.loads(b),8+n)
 for e in meta['weights'].values():
  assert index[e['checkpoint_key']]==e['shard']
  d,base=result[e['shard']];x=d[e['checkpoint_key']]
  assert x['dtype']==e['dtype']and x['shape']==e['shape']and x['data_offsets']==e['data_offsets']
  assert base+x['data_offsets'][0]==e['absolute_offset']and x['data_offsets'][1]-x['data_offsets'][0]==e['bytes']
 return root

def read_weight(root,e):
 with(root/e['shard']).open('rb')as f:f.seek(e['absolute_offset']);b=f.read(e['bytes'])
 assert len(b)==e['bytes']
 return b

def inspect(meta,root):
 config(meta);root=headers(meta,root);pins={};total=0
 for name,e in sorted(meta['weights'].items()):
  b=read_weight(root,e);pins[name]=dict(bytes=len(b),SHA256=H(b));total+=len(b)
 return dict(schema=1,layer=meta['layer'],weights=pins,total_selected_weight_bytes=total,original_config_SHA256=meta['original_config_SHA256'],original_index_SHA256=meta['original_index_SHA256'],header_SHA256={k:v['header_SHA256']for k,v in meta['headers'].items()},whole_shard_hashed=False,Native_dlopen=False,Runtime=False,GPU=False,weight_values_exported=False)

def load_checkpoint(trial,meta,pins,root,deadline=lambda stage:None):
 """Explicit device-phase caller only; owns a real Trial created on its owner thread."""
 start=time.monotonic();config(meta);root=headers(meta,root)
 assert set(pins['weights'])==set(meta['weights'])and pins['layer']==meta['layer']
 assert pins['original_config_SHA256']==meta['original_config_SHA256']and pins['original_index_SHA256']==meta['original_index_SHA256']
 validate_parameters(trial.read('parameters'),meta);loaded=[];total=0
 for name,e in sorted(meta['weights'].items()):
  deadline('checkpoint.read.'+name);b=read_weight(root,e);p=pins['weights'][name];assert len(b)==p['bytes']and H(b)==p['SHA256'],'original weight range bytes changed'
  # Synchronous Native load/drain returns before this buffer/source owner dies.
  buffer=C.create_string_buffer(b,len(b));deadline('checkpoint.load.'+name);trial.call('load',name.encode(),buffer,len(b));loaded.append(name);total+=len(b)
 deadline('checkpoint.postload');trial.call('postload')
 return dict(loaded=loaded,bytes=total,whole_checkpoint_load_wall_s=time.monotonic()-start,includes_hash_read_H2D_Native_load_and_postload=True,scope='Seven actual component parameters, not complete43layer Model load')
