"""Real rank0 checkpoint/scale indexing + F32 contraction diagnostic only.

This is not a GPU quantizer/BLAS emulator. It intentionally does NOT qualify
the reordered FP8 activation quantization, BF16 product rounding, device FTZ,
or complete attention. Those remain first-ON actual-device measurements.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def bf16(x):
 x=np.asarray(x,dtype=np.float32);u=x.view(np.uint32);return((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
def main():
 p=argparse.ArgumentParser();p.add_argument('--private-rank0',required=True);a=p.parse_args();root=Path(__file__).resolve().parent
 identity=json.loads((root/'SOURCE_IDENTITY.json').read_bytes());private=Path(a.private_rank0)
 for name,row in identity['private_payloads'].items():assert hashlib.sha256((private/name).read_bytes()).hexdigest()==row['SHA256']
 weight=np.fromfile(private/'kv_b-rank0-weight.u8',dtype=np.uint8).reshape(3584,512);scales=np.fromfile(private/'kv_b-global-scale.f32',dtype='<f4').reshape(224,4)
 assert np.isfinite(scales).all()and(scales>0).all()
 # Canonical safetensors F8_E4M3 representation. This does not substitute for
 # the active helper Decode/quantization/kernel-domain device qualification.
 code=np.arange(256,dtype=np.uint8);e=((code>>3)&15).astype(np.int32);m=(code&7).astype(np.float32)
 lut=np.where(e==0,m*np.float32(2**-9),(np.float32(1)+m/np.float32(8))*np.exp2((e-7).astype(np.float32))).astype(np.float32)
 lut=np.where((code&128)!=0,-lut,lut).astype(np.float32);lut[(code&127)==127]=np.nan
 decoded=lut[weight];assert np.isfinite(decoded).all()
 rows=np.arange(3584,dtype=np.int64);columns=np.arange(512,dtype=np.int64)
 fixed_weight=(decoded*scales[rows[:,None]//128,columns[None,:]//128]).astype(np.float32)
 value_rows=np.array([h*448+192+v for h in range(8)for v in range(256)],dtype=np.int64)
 assert len(set(map(int,value_rows)))==2048 and not np.array_equal(value_rows,np.arange(192,2240))
 wv=fixed_weight[value_rows].reshape(8,256,512)
 rng=np.random.default_rng(431038);latent=bf16(rng.standard_normal((2048,512),dtype=np.float32)*np.float32(.1))
 # A single fixed input; two probability concentration patterns, not a kernel grid.
 projected=np.matmul(latent,fixed_weight.T,dtype=np.float32).reshape(2048,8,448)[:,:,192:]
 results=[]
 for name,contrast in [('smooth',np.float32(1)),('peaked',np.float32(8))]:
  logits=rng.standard_normal((8,2048),dtype=np.float32)*contrast;logits-=logits.max(axis=1,keepdims=True)
  prob=np.exp(logits,dtype=np.float32);prob/=prob.sum(axis=1,keepdims=True,dtype=np.float32);prob=bf16(prob)
  old=np.stack([np.matmul(prob[h:h+1],projected[:,h,:],dtype=np.float32)[0]for h in range(8)])
  aggregate=np.matmul(prob,latent,dtype=np.float32)
  new=np.stack([np.matmul(aggregate[h:h+1],wv[h].T,dtype=np.float32)[0]for h in range(8)])
  rounded=np.stack([np.matmul(bf16(aggregate[h:h+1]),wv[h].T,dtype=np.float32)[0]for h in range(8)])
  def metrics(x):
   diff=(x-old).astype(np.float32);rms=np.sqrt(np.mean(old*old,dtype=np.float32),dtype=np.float32)
   return dict(finite=bool(np.isfinite(x).all()),NRMSE=float(np.sqrt(np.mean(diff*diff,dtype=np.float32),dtype=np.float32)/max(rms,np.float32(1e-20))),maxabs=float(np.max(np.abs(diff))))
  results.append(dict(pattern=name,F32_linear_reassociation=metrics(new),aggregate_BF16_rounding_only=metrics(rounded)))
 # All TP8 V scale addresses, including head boundaries crossing row128.
 addresses=[]
 for rank in range(8):
  for h in range(8):
   for v in range(256):
    row=rank*3584+h*448+192+v
    for b in range(4):assert 0<=(row//128)*4+b<896;addresses.append((row//128)*4+b)
 q=dict(completed=True,passed=all(x['F32_linear_reassociation']['finite']and x['aggregate_BF16_rounding_only']['finite']for x in results),
  source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),actual_rank0_checkpoint_slice_verified=True,
  checkpoint_global_weight_shape=[28672,512],rank0_weight_shape=[3584,512],global_scale_shape=[224,4],
  scale_min=float(scales.min()),scale_max=float(scales.max()),all_rank_value_scale_addresses_checked=len(addresses),
  head_value_rows='rank*3584+head*448+192+[0..255], then floor(globalrow/128)*4+Kblock',
  M8_output_diagonal_flat_rows=[h*8+h for h in range(8)],results=results,math_dtype='float32; BF16 RNE simulated explicitly',
  activation_FP8_quantization_emulated=False,Native=False,GPU=0,Model_READY=False,
  scope='Index/algebra diagnostic on real checkpoint; not current vendor finite-precision correctness, latency, or task quality',
  unqualified=['baseline per-row FP8 activation quantization','new aggregate M8 FP8 quantization','GPU block/GEMM accumulation association','complete QK/softmax/attention/O/TP','state/owner/device lifecycle'])
 (root/'algebra-CPU-result.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
