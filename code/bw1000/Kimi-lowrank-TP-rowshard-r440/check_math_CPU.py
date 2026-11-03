"""CPU mathematical layout reference only, not SDK/GPU or performance proof."""
import time
started=time.monotonic()
import torch
def check():
 assert not torch.cuda.is_initialized()
 torch.set_num_threads(4)
 rows=[]
 for n,k in [(3584,7168),(7168,3584)]:
  # Finite synthetic BF16 weights and input, independently evaluated in F32.
  # No checkpoint or Model activation is loaded and no F64 is used.
  w=((torch.arange(n*k,dtype=torch.float32).reshape(n,k)%251)-125)/128
  w=w.bfloat16().float()
  x=(((torch.arange(k,dtype=torch.float32)%29)-14)/16).bfloat16().float()
  expected=torch.mv(w,x)
  shards=[torch.mv(w.narrow(0,rank*(n//16),n//16),x)for rank in range(16)]
  gathered=torch.cat(shards)
  assert gathered.shape==expected.shape==(n,)
  difference=gathered-expected
  nr=float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(expected))
  assert bool(torch.isfinite(gathered).all())and nr<2e-6
  reversed_order=torch.cat(list(reversed(shards)))
  assert not torch.allclose(reversed_order,expected,rtol=2e-5,atol=2e-4),'Reference cannot distinguish wrong TP order'
  rounded_expected=expected.bfloat16().float();rounded_gathered=gathered.bfloat16().float()
  bf16_nr=float(torch.linalg.vector_norm(rounded_gathered-rounded_expected)/torch.linalg.vector_norm(rounded_expected))
  rows.append({'N':n,'K':k,'TP':16,'M':1,'local_N':n//16,'F32_reconstruction_NRMSE':nr,'F32_max_abs':float(difference.abs().max()),'BF16_reconstruction_NRMSE':bf16_nr,'BF16_different_values_diagnostic':int((rounded_gathered!=rounded_expected).sum()),'wrong_order_rejected':True,'weight_bytes_replicated_per_rank':n*k*2,'logical_weight_slice_bytes_per_rank':n*k*2//16,'allgather_output_bytes_per_rank':n*2})
 assert not torch.cuda.is_initialized()
 return {'passed':True,'cases':rows,'CPU_reference_s':time.monotonic()-started,'GPU_execution':False,'Model':False,'checkpoint':False,'CUDA_initialized':False,'performance_gain_qualified':False,'scope':'CPU F32 evaluation of syntheticBF16 row partition/reconstruction; not device rounding/collective/owner/input replication proof.'}
if __name__=='__main__':
 import json
 print(json.dumps(check()))
