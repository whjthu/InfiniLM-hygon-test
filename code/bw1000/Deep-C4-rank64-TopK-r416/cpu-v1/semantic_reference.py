"""Actual Torch CPU out-reference; Native/GPU/weights are never opened."""
from pathlib import Path
import argparse,json,hashlib,subprocess
import numpy as np
N,K,PAD=2124,512,4096

def keys(bits):
 b=np.asarray(bits,np.uint32);a=b&np.uint32(0x7fffffff)
 out=np.where((b&np.uint32(0x80000000))!=0,~b,b^np.uint32(0x80000000)).astype(np.uint32)
 out[a==0]=np.uint32(0x80000000);out[a>np.uint32(0x7f800000)]=np.uint32(0xffffffff)
 return out

def network(bits):
 k=keys(bits).astype(np.uint64);rank=np.zeros(PAD,np.uint64);rank[:N]=(k<<np.uint64(32))|(np.uint64(0xffffffff)-np.arange(N,dtype=np.uint64))
 for span in [1<<i for i in range(1,13)]:
  stride=span>>1
  while stride:
   for i in range(PAD):
    j=i^stride
    if j<=i:continue
    if (rank[j]>rank[i])if(i&span)==0 else(rank[i]>rank[j]):rank[i],rank[j]=rank[j],rank[i]
   stride>>=1
 assert np.all(rank[:K]>0)
 return (np.uint64(0xffffffff)-(rank[:K]&np.uint64(0xffffffff))).astype(np.int64)

def cases():
 rng=np.random.default_rng(4158496);spec=[]
 def add(name,b):
  b=np.asarray(b,np.uint32).reshape(-1);assert b.size==N;spec.append((name,b.copy()))
 add('random_F32',rng.normal(size=N).astype(np.float32).view(np.uint32))
 add('all_equal',np.full(N,np.float32(1)).view(np.uint32))
 add('all_negative_infinity_padding_must_lose',np.full(N,np.uint32(0xff800000)))
 add('all_positive_infinity',np.full(N,np.uint32(0x7f800000)))
 add('all_nan_payload_signaling_signs',np.resize(np.array([0x7fc00001,0xffc00002,0x7f800001,0xff800001],np.uint32),N))
 add('signed_zero_tie',np.resize(np.array([0,0x80000000],np.uint32),N))
 b=np.zeros(N,np.float32);b[:511]=2;b[511:611]=1;add('cutoff_512_ties',b.view(np.uint32))
 b=np.full(N,-np.inf,np.float32);b[-1]=-3;add('real_negative_inf_better_than_invalid_padding',b.view(np.uint32))
 b=np.full(N,-2,np.float32);b[:511]=2;b[511]=1;b[512]=np.nextafter(np.float32(1),np.float32(0));b[513]=np.nextafter(np.float32(1),np.float32(2));add('cutoff_oneULP',b.view(np.uint32))
 add('ascending',np.arange(N,dtype=np.float32).view(np.uint32));add('descending',np.arange(N,0,-1,dtype=np.float32).view(np.uint32))
 b=rng.integers(0,2**32,N,dtype=np.uint32);b[:16]=np.array([0,0x80000000,0x7f800000,0xff800000,0x7fc00001,0xffc00002,0x7f800001,0xff800001,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f7fffff,0xff7fffff],np.uint32);add('adversarial_float_bits',b)
 b=np.zeros(N,np.uint32);b[:600]=np.uint32(0x7fc00001);add('nan_cutoff_tie',b)
 add('positive_negative_smallest_subnormals',np.resize(np.array([1,0x80000001,2,0x80000002,0,0x80000000],np.uint32),N))
 return spec

def validate(bits,ids,value_bits,expected_keys,require_original_bits):
 assert ids.shape==(K,)and len(set(ids.tolist()))==K and np.all((ids>=0)&(ids<N))
 k=keys(bits);assert np.array_equal(k[ids],expected_keys),'wrong ranked value set/order'
 if require_original_bits:assert np.array_equal(value_bits,bits[ids]),'values must bit-copy original score[id]'
 else:assert np.array_equal(keys(value_bits),k[ids]),'Torch returned values not equivalent to selected inputs'
 cutoff=expected_keys[-1];strict=np.flatnonzero(k>cutoff);assert set(strict.tolist())<=set(ids.tolist());assert np.all(k[ids]>=cutoff)

def run(folder,exe=None):
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();folder=Path(folder);folder.mkdir(exist_ok=True)
 result=[]
 for name,bits in cases():
  # CPU tensor and out= tuple exercise the same Torch operation semantics, not a mock.
  t=torch.from_numpy(bits.view(np.float32).copy()).reshape(1,N);v=torch.empty((1,K),dtype=torch.float32,device='cpu');i=torch.empty((1,K),dtype=torch.int64,device='cpu');torch.topk(t,K,dim=1,largest=True,sorted=True,out=(v,i))
  expected=np.sort(keys(bits))[::-1][:K];validate(bits,i.numpy().reshape(-1),v.numpy().reshape(-1).view(np.uint32),expected,False)
  own=network(bits);validate(bits,own,bits[own],expected,True);rawkeys=keys(bits);assert np.array_equal(own,np.asarray(sorted(range(N),key=lambda j:(-int(rawkeys[j]),j))[:K],np.int64))
  if exe:
   p=folder/(name+'.input.bin');p.write_bytes(bits.tobytes());o=folder/(name+'.result.bin');subprocess.run([str(exe),'--sort',str(p),str(o)],check=True,timeout=15,capture_output=True);raw=o.read_bytes();assert len(raw)==K*12;vals=np.frombuffer(raw[:K*4],np.uint32);idx=np.frombuffer(raw[K*4:],np.int64);validate(bits,idx,vals,expected,True);assert np.array_equal(idx,own)
  result.append(dict(case=name,input_SHA256=hashlib.sha256(bits.tobytes()).hexdigest(),Torch_CPU_out_reference=True,ranked_value_set_and_cutoff=True,valid_unique_indices=True,stable_lowID_tie_legal=True,padding_never_selected=True,values_equal_original_bits=True,actual_shared_CXX_comparator_and_network=bool(exe)))
 assert not torch.cuda.is_initialized()
 return dict(passed=True,CPU_only=True,GPU_initialized=False,Native_dlopen=False,torch_version=torch.__version__,cases=result,case_count=len(result),actual_CXX_consumer=bool(exe),NaN_highest_equivalence_group=True,signed_zero_equal_group=True,padding_below_real_negative_infinity=True,no_GPU_F64=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--exe');x=p.parse_args();print(json.dumps(run(x.folder,x.exe)))
