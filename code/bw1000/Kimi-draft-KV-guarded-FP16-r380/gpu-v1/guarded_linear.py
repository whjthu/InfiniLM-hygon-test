"""DefaultOFF private FP16 compute copy, original BF16 Linear stays immutable."""
import torch
import torch.nn.functional as F
LIMIT=65504.0
def version(t):
 try:return t._version
 except RuntimeError:return None
class GuardedFP16Linear(torch.nn.Module):
 def __init__(self,original):
  super().__init__();assert isinstance(original,torch.nn.Linear)and original.bias is None and original.weight.dtype==torch.bfloat16 and not original.weight.requires_grad
  self.original=original;w=original.weight;self.signature=(id(w),w.data_ptr(),tuple(w.shape),w.device,w.dtype,version(w));self.counts=dict(baseline=0,hit=0,range_fallback=0,weight_range_fallback=0,output_fallback=0,rejected_nonfinite=0);self.last_path=None
  if not bool(torch.isfinite(w).all().item()):raise ValueError('nonfinite weight')
  self.weight_maxabs=float(w.abs().amax().item());self.weight16=w.to(dtype=torch.float16,copy=True)if self.weight_maxabs<=LIMIT else None
 def forward(self,x,*,enabled=False):
  w=self.original.weight
  if (id(w),w.data_ptr(),tuple(w.shape),w.device,w.dtype,version(w))!=self.signature:raise RuntimeError('weight identity/version changed')
  if not enabled:self.counts['baseline']+=1;self.last_path='baseline';return self.original(x)
  if x.dtype!=torch.bfloat16 or x.device!=w.device or x.shape[-1]!=w.shape[-1]:raise ValueError('input dtype/device/shape')
  if not bool(torch.isfinite(x).all().item()):self.counts['rejected_nonfinite']+=1;self.last_path='rejected_nonfinite';raise ValueError('nonfinite input rejected')
  # .item -> Pythonfloat comparison is deliberate: BF16 scalar65504 rounds65536.
  maxabs=float(x.abs().amax().item())
  if maxabs>LIMIT:self.counts['range_fallback']+=1;self.last_path='range_fallback';return self.original(x)
  if self.weight16 is None:self.counts['weight_range_fallback']+=1;self.last_path='weight_range_fallback';return self.original(x)
  half=x.to(torch.float16);value=F.linear(half,self.weight16)
  # Even finite inputs/weights can overflow an FP16 output after accumulation.
  if not bool(torch.isfinite(value).all().item()):self.counts['output_fallback']+=1;self.last_path='output_fallback';return self.original(x)
  result=value.to(torch.bfloat16);self.counts['hit']+=1;self.last_path='hit';return result
