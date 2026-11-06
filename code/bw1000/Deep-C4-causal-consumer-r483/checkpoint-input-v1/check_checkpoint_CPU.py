"""Metadata-only CPU regression; no weights, Native, Torch, SDK or device."""
import copy,json,math
from pathlib import Path
from checkpoint import config,validate_parameters,SIZES
m=json.loads((Path(__file__).parent/'CHECKPOINT_METADATA.json').read_bytes())
actual=[dict(name=n,shape=e['shape'],dtype=m['dtype_codes'][e['dtype']],bytes=e['bytes'],contiguous=True,loaded=False)for n,e in sorted(m['weights'].items())]
config(m);validate_parameters(actual,m)
assert sum(e['bytes']for e in m['weights'].values())==13144320
assert all(e['checkpoint_key'].startswith('layers.2.attn.indexer.')for e in m['weights'].values())
assert m['dtype_codes']=={'I8':0,'BF16':9,'F32':10}
count=0

def reject(f):
 global count
 try:f()
 except AssertionError:count+=1;return
 raise AssertionError('unsafe metadata accepted')

reject(lambda:validate_parameters(actual[:-1],m))
x=copy.deepcopy(actual);x.append(copy.deepcopy(x[0]));reject(lambda:validate_parameters(x,m))
x=copy.deepcopy(actual);x[1]=copy.deepcopy(x[0]);reject(lambda:validate_parameters(x,m))
for name,field,value in [('compressor.norm.weight','shape',[512]),('compressor.ape','shape',[4,1024]),('compressor.wkv.weight','dtype',8),('wq_b.scale','dtype',9),('wq_b.weight','bytes',1),('weights_proj.weight','loaded',True),('compressor.wgate.weight','contiguous',False)]:
 x=copy.deepcopy(actual);next(a for a in x if a['name']==name)[field]=value;reject(lambda:validate_parameters(x,m))
x=copy.deepcopy(m);x['config_runtime']['rope_scaling']['factor']=1;reject(lambda:config(x))
x=copy.deepcopy(m);x['config_runtime']['max_position_embeddings']=16384;reject(lambda:config(x))
assert count==12
print('PASS: exact seven checkpoint keys/13,144,320 bytes, 12 rejection cases including D512/D128 mixup, dtype codes, full32768 and unchanged RoPE; no Native/weights/device')
