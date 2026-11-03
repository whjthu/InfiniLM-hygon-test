"""Independent Torch CPU specification; no Native, Runtime, weights or GPU calls."""
from pathlib import Path
import numpy as np
import json,hashlib,subprocess,argparse
N,K=256,6

def value_keys(a):
 b=np.asarray(a,np.float32).view(np.uint32);v=np.where(b&np.uint32(0x80000000),~b,b^np.uint32(0x80000000));ab=b&np.uint32(0x7fffffff);v[ab==0]=np.uint32(0x80000000);v[ab>np.uint32(0x7f800000)]=np.uint32(0xffffffff);return v

def numpy_suffix(logits,bias):
 with np.errstate(all='ignore'):
  soft=np.where(logits>20,logits,np.log1p(np.exp(logits,dtype=np.float32),dtype=np.float32));scores=np.sqrt(soft,dtype=np.float32);selection=np.add(scores,bias,dtype=np.float32)
  keys=value_keys(selection);ids=np.asarray(sorted(range(N),key=lambda i:(-int(keys[i]),i))[:K],np.int64);v=scores[ids]
  total=np.float32(np.float32(np.float32(v[0]+v[1])+np.float32(v[2]+v[3]))+np.float32(v[4]+v[5]));denom=np.float32(total+np.float32(1e-20));weights=np.multiply(np.divide(v,denom,dtype=np.float32),np.float32(1.5),dtype=np.float32)
 return dict(scores=scores,selection=selection,ids=ids,weights=weights)

def cases():
 rng=np.random.default_rng(419);out=[]
 def add(name,x,b=None):out.append((name,np.asarray(x,np.float32).reshape(N),np.zeros(N,np.float32)if b is None else np.asarray(b,np.float32).reshape(N)))
 add('normal_random',rng.normal(0,4,N),rng.normal(0,.2,N))
 add('large_positive_logits',np.linspace(50,100,N))
 add('threshold20_neighbors',np.resize(np.array([np.nextafter(np.float32(20),np.float32(0)),20,np.nextafter(np.float32(20),np.float32(100)),19.99,20.01],np.float32),N),np.linspace(-.1,.1,N))
 add('all_negative_infinity',np.full(N,-np.inf))
 add('all_positive_infinity',np.full(N,np.inf))
 x=rng.normal(size=N).astype(np.float32);x[:12]=np.nan;add('NaN_logits',x)
 add('all_equal_zero_logits',np.zeros(N))
 add('signed_zero_logits_and_bias',np.resize(np.array([0.,-0.],np.float32),N),np.resize(np.array([-0.,0.],np.float32),N))
 bias=np.zeros(N,np.float32);bias[:5]=2;add('cutoff_exact_ties',np.zeros(N),bias)
 bias=np.linspace(-2,0,N,dtype=np.float32);bias[:6]=3;add('margin_separated',np.linspace(-1,1,N),bias)
 x=rng.normal(0,2,N).astype(np.float32);scores=numpy_suffix(x,np.zeros(N,np.float32))['scores'];add('rounding_sensitive_cancellation',x,-scores)
 add('negative_underflow',np.linspace(-110,-80,N),rng.normal(size=N))
 add('bias_selects_low_unbiased_scores',np.linspace(-15,15,N),np.linspace(20,-20,N))
 add('skewed_selected_normalization',np.linspace(-80,80,N),np.resize(np.array([20,-10,0,3],np.float32),N))
 return out

def torch_reference(torch,logits,bias):
 x=torch.from_numpy(logits.copy()).reshape(1,N);b=torch.from_numpy(bias.copy()).reshape(1,N)
 scores=torch.sqrt(torch.nn.functional.softplus(x,beta=1,threshold=20));selection=scores+b;values,ids=torch.topk(selection,K,dim=1,largest=True,sorted=True)
 chosen=torch.gather(scores,1,ids);weights=chosen/(torch.sum(chosen,dim=1,keepdim=True)+torch.tensor(1e-20,dtype=torch.float32));weights=weights*torch.tensor(1.5,dtype=torch.float32)
 return dict(scores=scores.numpy().reshape(N),selection=selection.numpy().reshape(N),ids=ids.numpy().reshape(K),weights=weights.numpy().reshape(K))

def validate(torch,logits,bias,reference,candidate):
 ids=candidate['ids'];assert ids.shape==(K,)and len(set(ids.tolist()))==K and np.all((ids>=0)&(ids<N))
 for key in ['scores','selection']:
  x,y=candidate[key],reference[key];assert x.shape==y.shape and np.array_equal(np.isnan(x),np.isnan(y))and np.array_equal(np.isposinf(x),np.isposinf(y))and np.array_equal(np.isneginf(x),np.isneginf(y)),key
  finite=np.isfinite(x);error=float(np.max(np.abs(x[finite].astype(np.float64)-y[finite].astype(np.float64))))if finite.any()else 0.;bound=max(2e-6,5e-7*float(np.max(np.abs(y[finite])))if finite.any()else 0.);assert error<=bound,(key,error,bound)
 keys=value_keys(candidate['selection']);assert np.array_equal(keys[ids],np.sort(keys)[::-1][:K]),'candidate wrong ranked selection set/order'
 refkeys=value_keys(reference['selection']);ordered=np.sort(reference['selection'])[::-1]
 finite=np.isfinite(reference['selection'])&np.isfinite(candidate['selection']);error=float(np.max(np.abs(reference['selection'][finite].astype(np.float64)-candidate['selection'][finite].astype(np.float64))))if finite.any()else 0.
 margin=float(ordered[5]-ordered[6])if np.isfinite(ordered[[5,6]]).all()else None
 separated=margin is not None and margin>2*error
 same=set(ids.tolist())==set(reference['ids'].tolist())
 if separated:assert same,('margin-separated wrong routes',margin,error)
 # Independent normalization at the candidate's actual IDs; no bias in gathered values.
 chosen=torch.from_numpy(reference['scores'][ids].copy()).reshape(1,K);expected=(chosen/(torch.sum(chosen,dim=1,keepdim=True)+torch.tensor(1e-20,dtype=torch.float32))*torch.tensor(1.5,dtype=torch.float32)).numpy().reshape(K)
 actual=candidate['weights'];assert np.array_equal(np.isnan(actual),np.isnan(expected))and np.array_equal(np.isposinf(actual),np.isposinf(expected))and np.array_equal(np.isneginf(actual),np.isneginf(expected))
 good=np.isfinite(expected);ma=float(np.max(np.abs(actual[good].astype(np.float64)-expected[good].astype(np.float64))))if good.any()else 0.;assert ma<2e-6,ma
 nr=float(np.linalg.norm(actual[good].astype(np.float64)-expected[good].astype(np.float64))/(np.linalg.norm(expected[good].astype(np.float64))+1e-30))if good.any()else 0.;assert nr<2e-6,nr
 return dict(score_and_selection_classification=True,selection_maxabs_error=error,baseline_cutoff_margin=margin,margin_separated=separated,same_expert_set=same,route_ID_difference_ambiguous_not_promoted=not same,weights_maxabs=ma,weights_NRMSE=nr,finite_weights=bool(np.isfinite(actual).all()),nonfinite_matches_original_classification=True)

def run(folder,exe=None):
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();folder=Path(folder);folder.mkdir(exist_ok=True);rows=[]
 for name,logits,bias in cases():
  ref=torch_reference(torch,logits,bias);candidate=numpy_suffix(logits,bias)
  if exe:
   p=folder/(name+'.input');o=folder/(name+'.output');p.write_bytes(np.concatenate([logits,bias,np.array([1e-20,1.5],np.float32)]).tobytes());subprocess.run([str(exe),'--suffix',str(p),str(o)],check=True,capture_output=True,timeout=20);b=o.read_bytes();assert len(b)==2120
   candidate=dict(ids=np.frombuffer(b[:48],np.int64).copy(),weights=np.frombuffer(b[48:72],np.float32).copy(),scores=np.frombuffer(b[72:1096],np.float32).copy(),selection=np.frombuffer(b[1096:],np.float32).copy())
  result=validate(torch,logits,bias,ref,candidate);rows.append(dict(case=name,input_SHA256=hashlib.sha256(logits.tobytes()+bias.tobytes()).hexdigest(),**result))
 # Wrong well-separated expert routes are a hard failure, not legal-tie tolerance.
 logits=np.linspace(-2,2,N,dtype=np.float32);bias=np.zeros(N,np.float32);ref=torch_reference(torch,logits,bias);bad=numpy_suffix(logits,bias);bad['ids'][0]=0
 try:validate(torch,logits,bias,ref,bad)
 except AssertionError:pass
 else:raise AssertionError('wrongroute negative accepted')
 assert not torch.cuda.is_initialized()
 return dict(passed=True,CPU_only=True,Native_dlopen=False,GPU_initialized=False,Torch_CPU_reference=True,actual_shared_CXX_math=bool(exe),case_count=len(rows),cases=rows,margin_separated_wrongroute_rejected=True,F32_sum_tree='((v0+v1)+(v2+v3))+(v4+v5),then+epsilon; allFloat32; not claimedbitexactATenGPU',nonfinite_cases_are_behavioral_not_eligible_Model_fixtures=True,strict_Model_quality=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--exe');a=p.parse_args();print(json.dumps(run(a.folder,a.exe)))
