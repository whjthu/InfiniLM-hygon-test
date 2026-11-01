"""Small real HC parameter tiles plus disclosed synthetic T1 residuals."""
import pathlib,json,hashlib,numpy as np
from reference import bf16_rn,full_pre
MODEL=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8')
GROUPS=((0,'attn'),(0,'ffn'),(20,'attn'),(20,'ffn'))
SEEDS=(781,1889)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def load(rank):
 assert type(rank)is int and 0<=rank<8
 import torch
 from safetensors import safe_open
 cfg=json.loads((MODEL/'config.json').read_bytes());index=MODEL/'model.safetensors.index.json';mapping=json.loads(index.read_bytes())['weight_map'];assert cfg['hidden_size']==4096 and cfg['hc_eps']>0 and 1<=cfg['hc_sinkhorn_iters']<=128
 layer,kind=GROUPS[rank//2];seed=SEEDS[rank%2];arrays=[];proof=[]
 for name,shape in [('fn',[24,16384]),('base',[24]),('scale',[3])]:
  key=f'layers.{layer}.hc_{kind}_{name}';assert key in mapping
  with safe_open(MODEL/mapping[key],framework='pt',device='cpu')as reader:t=reader.get_tensor(key).contiguous()
  assert t.dtype==torch.float32 and list(t.shape)==shape
  a=t.numpy().copy();assert np.isfinite(a).all();arrays.append(a);proof.append(dict(key=key,shape=shape,dtype='F32',SHA256=hashlib.sha256(a.tobytes()).hexdigest(),shard=mapping[key]))
 rng=np.random.default_rng(seed);x=rng.standard_normal((1,4,4096),dtype=np.float32)*np.float32(.5 if rank%2==0 else 2.)
 bits=bf16_rn(x);expected=full_pre(bits,*arrays,float(cfg['rms_norm_eps']),float(cfg['hc_eps']),int(cfg['hc_sinkhorn_iters']))
 data=dict(input_bits=bits,weights=arrays,expected=expected,rms_eps=float(cfg['rms_norm_eps']),hc_eps=float(cfg['hc_eps']),iterations=int(cfg['hc_sinkhorn_iters']))
 meta=dict(rank=rank,layer=layer,kind=kind,seed=seed,shape=[1,4,4096],input_kind='Synthetic residual, not actual8K activation or service workload',input_SHA256=hashlib.sha256(bits.tobytes()).hexdigest(),weights=proof,config_SHA256=sha(MODEL/'config.json'),index_SHA256=sha(index),checkpoint_scope='Only three HC parameter tiles; no targetModel construction/loading',CPU_reference_output_SHA256={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in expected.items()})
 return data,meta
