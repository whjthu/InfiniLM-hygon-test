import pathlib,json,hashlib,ast,math,types,sys,typing,functools,__future__
import torch
import torch.nn as nn
import torch.nn.functional as F
def selected(path,names,env):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names];assert {n.name for n in nodes}==set(names)
 class Eager(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='torchspec.models.draft.dspark' and [a.name for a in n.names]==['K3DSparkMLAAttention']:return ast.Pass()
   return self.generic_visit(n)
  def visit_FunctionDef(self,n):
   n.decorator_list=[x for x in n.decorator_list if not isinstance(x,ast.Call)or ast.unparse(x.func)!='torch.compile'];return self.generic_visit(n)
 tree=Eager().visit(ast.Module(body=nodes,type_ignores=[]));ast.fix_missing_locations(tree);exec(compile(tree,str(path),'exec',flags=__future__.annotations.compiler_flag),env)
def make_factory(config,source_dir):
 inc=pathlib.Path(source_dir)
 pins={'r9000_torchspec_deepseek_eagle.py': '56d8c18321bc270493d90efaa32f9aabfe00499c12ed96e7c12359200decb580', 'r9000_torchspec_dflash.py': '81489dec80b0d4075ed59ac4e48bc12382f3d50e880d1ebdd2dfbfe702a831d7', 'r9000_torchspec_dflash2.py': '195dfadcab1c3c4fb2fdc76428dc5b19fa6af2e2358ba3b9b58c38d5d9f03fad', 'r9000_torchspec_dspark.py': '0794d502e5be22cacd7bdc27cdf128107a1d35b2303da8de204a1422b6eb6382', 'r9000_torchspec_llama3_eagle.py': '5c94ee86ab8f8b680b473696706a8bacbf1bfb398342f86193786844acf60f3e'}
 for name,h in pins.items():
  if hashlib.sha256((inc/name).read_bytes()).hexdigest()!=h:raise ValueError('creator source hash mismatch '+name)
 env=dict(torch=torch,nn=nn,F=F,math=math,partial=functools.partial,Optional=typing.Optional,Tuple=typing.Tuple,PretrainedConfig=types.SimpleNamespace,DFlashConfig=types.SimpleNamespace,DFlash2Config=types.SimpleNamespace,K3DSparkConfig=types.SimpleNamespace,resolve_rope_theta=lambda c:c.rope_parameters['rope_theta'])
 selected(inc/'r9000_torchspec_llama3_eagle.py',['LlamaRotaryEmbedding','LlamaYarnRotaryEmbedding','yarn_find_correction_dim','yarn_find_correction_range','yarn_get_mscale','yarn_linear_ramp_mask','rope_config_get','build_rotary_embedding'],env)
 selected(inc/'r9000_torchspec_deepseek_eagle.py',['_rotate_half_interleaved','DeepSeekMLAAttention'],env)
 selected(inc/'r9000_torchspec_dflash.py',['DFlashRMSNorm','DFlashMLP'],env)
 def cpu_flex(*,query,key,value,block_mask,scale):return F.scaled_dot_product_attention(query,key,value,attn_mask=block_mask,dropout_p=0.,is_causal=False,scale=scale)
 env['compile_friendly_flex_attention']=cpu_flex;selected(inc/'r9000_torchspec_dspark.py',['_apply_rope_interleaved','K3DSparkMLAAttention'],env);env['DFlashAttention']=env['K3DSparkMLAAttention'];selected(inc/'r9000_torchspec_dflash.py',['DFlashDecoderLayer'],env)
 selected(inc/'r9000_torchspec_dflash2.py',['DFlashGroupedConv','CandidateSelector','DFlash2DecoderLayer'],env)
 rope=config['rope_parameters'];nested=config['dflash_config'];cfg=types.SimpleNamespace(**dict(config,rope_scaling=rope,attention_mode=nested['attention_mode'],block_size=8,conv_kernel_size=2,conv_group_size=16))
 assert cfg.num_hidden_layers==5 and cfg.fc_norm is False and nested.get('input_embedding_scale',1.)==1.
 class Draft(nn.Module):
  def __init__(self):
   super().__init__();h=cfg.hidden_size;self.config=cfg;self.mask_token_id=nested['mask_token_id'];self.embed_tokens=nn.Embedding(cfg.vocab_size,h);self.context_proj=nn.Linear(5*h,h,bias=False);self.context_norm=env['DFlashRMSNorm'](h,cfg.rms_norm_eps);self.final_norm=env['DFlashRMSNorm'](h,cfg.rms_norm_eps);self.layers=nn.ModuleList([env['DFlash2DecoderLayer'](cfg)for _ in range(5)]);self.candidate_selector=env['CandidateSelector'](h,cfg.vocab_size,nested['selector_rank'],nested['selector_top_k'])
 return Draft
