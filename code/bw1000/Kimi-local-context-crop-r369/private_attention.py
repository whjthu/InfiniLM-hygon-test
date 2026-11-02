"""Private read-only borrowed-weight call; sole arithmetic-source delta is RoPE span."""
import ast,copy,hashlib,__future__
from functools import lru_cache
from pathlib import Path
import torch

PIN='0794d502e5be22cacd7bdc27cdf128107a1d35b2303da8de204a1422b6eb6382'

@lru_cache(maxsize=1)
def private_forward(original):
 p=Path(original.__code__.co_filename)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==PIN
 tree=ast.parse(p.read_text())
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='K3DSparkMLAAttention')
 node=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='forward'))
 assert not node.decorator_list
 node.args.kwonlyargs.append(ast.arg(arg='absolute_span'));node.args.kw_defaults.append(None)
 edits=0
 for n in ast.walk(node):
  if isinstance(n,ast.Call)and ast.unparse(n.func)=='self.rotary_emb':
   kw=next(k for k in n.keywords if k.arg=='seq_len')
   assert ast.unparse(kw.value)=='total_len'
   kw.value=ast.Name(id='absolute_span',ctx=ast.Load());edits+=1
 assert edits==1
 tree=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
 env=dict(original.__globals__)
 exec(compile(tree,str(p)+'::r369_absolute_span','exec',flags=__future__.annotations.compiler_flag),env)
 return env['forward']

class SpanAttention(torch.nn.Module):
 def __init__(self,original,span):
  super().__init__();self.original=original;self.span=span
  assert type(span)is int and 1<=span<=32768
  self.function=private_forward(type(original).forward)
 def forward(self,*args,**kwargs):
  return self.function(self.original,*args,**kwargs,absolute_span=self.span)

def window_layer(layer,x,ctx,dp,cp,mask,absolute_span):
 # Copy Python module routing only; all parameters/buffers are borrowed refs.
 # Original model/layer/attention modules and state dictionaries are unmodified.
 local=copy.copy(layer);local._modules=dict(layer._modules)
 local.self_attn=SpanAttention(layer.self_attn,absolute_span)
 return local(x,ctx,dp,cp,mask)
