import pathlib,json,hashlib
import numpy as np
r=pathlib.Path.cwd();s=(r/'Deep-latent-prefill-private-r8936/deep_latent_prefill_tiles_r8933.cpp').read_text();old=next(v['text']for v in json.loads((r/'Deep-actual-latent-sources-r8932.json').read_bytes())['units']if v['source'].endswith('deepseek_v4_latent_model_r685.cpp'))
def function(t):
 a=t.index('rt::Tensor latent_attention(');b=t.index('{',a);depth=1;i=b+1
 while depth:
  depth+=int(t[i]=='{')-int(t[i]=='}');i+=1
 return t[a:i]
base=function(old);candidate=function(s);candidate='\n'.join(l for l in candidate.split('\n')if 'if(prefill_tiles_mode.load()'not in l);assert candidate==base
cases=[]
for n in [1,7,8,31,32,63,64,65,71,72,95,96,127,128,2048,4096,4097]:
 eligible=64<n<=4096 and(n%32==0 or n%32>=8);tiles=[(b,min(32,n-b))for b in range(0,n,32)]if eligible else[]
 if tiles:assert min(v for b,v in tiles)>=8 and sum(v for b,v in tiles)==n and [i for b,w in tiles for i in range(b,b+w)]==list(range(n))
 for b,w in tiles:assert b*8*512*2+w*8*512*2<=n*8*512*2
 cases.append(dict(tokens=n,tiled=eligible,tiles=len(tiles),tail=tiles[-1][1]if tiles else None))
# Row tiling does not renumber global banks/selections or create cross-query softmax.
rng=np.random.default_rng(8933);q=rng.normal(size=(95,2,8)).astype(np.float32);bank=rng.normal(size=(13,8)).astype(np.float32);ids=np.array([[(i+j*3)%13 for j in range(5)]for i in range(95)]);sink=np.array([.2,-.1],np.float32)
def f(q,ids):
 vals=bank[ids];score=np.einsum('thd,tkd->thk',q,vals,dtype=np.float32);score=np.concatenate([score,np.broadcast_to(sink[None,:,None],(len(q),2,1))],axis=2);z=np.exp(score-score.max(2,keepdims=True));p=z/z.sum(2,keepdims=True);return np.einsum('thk,tkd->thd',p[:,:,:-1],vals,dtype=np.float32)
a=f(q,ids);b=np.concatenate([f(q[i:i+32],ids[i:i+32])for i in range(0,len(q),32)]);assert np.array_equal(a,b)
out=dict(hardware='BW1000',level='CPU_TILE_CONSUMER',status='PASSED',CPU_only=True,noGPU_dlopen=True,source_SHA256=hashlib.sha256(s.encode()).hexdigest(),original_dispatch_body_unchanged_except_optin_prefill_branch=True,decode_predicate_unchanged=True,tail_cases=cases,global_KV_selection_sink_row_tiling_CPUoracle_equal=True,output_byte_offsets_partition=True,lifetime='wholeout/work/flag live entirehelper;query/selection/lens views live through each originalfused sync;workreused onlyafterthatdrain;newbranchcanonicalDeviceGuard restorescaller;originalcachecommitoutsideuntouched',scope='CPU partition/reference only, notGPU r681numericqualification or Modelquality. No intermediatebitexactacceptancegate.')
p=r/'Deep-latent-prefill-CPU-contract-r8936.json';p.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))