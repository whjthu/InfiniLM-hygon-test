"""CPU value comparison of the exact old/new native artifacts whose bytes differ."""
import hashlib,json,math,os,signal,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;signal.alarm(600)
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
import torch
torch.set_num_threads(2);assert not torch.cuda.is_initialized()
index=json.loads((ROOT/'PAIR_INDEX.json').read_bytes());rows=[];started=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for raw in iter(lambda:f.read(1024*1024),b''):h.update(raw)
 return h.hexdigest()
for pair in index['pairs']:
 paths=[]
 for side in ('old','new'):
  path=Path(index[side+'_root'])/'capture'/('rank-%02d'%pair['rank'])/pair['name']
  assert path.is_file() and path.stat().st_size==pair[side]['bytes'] and sha(path)==pair[side]['SHA256']
  paths.append(path)
 a,b=(torch.load(p,map_location='cpu',weights_only=True) for p in paths)
 row={'rank':pair['rank'],'name':pair['name'],'tensors_checked':0,'different_tensors':0,'different_primitive_values':0,'structure_differences':0,'stride_differences':0,'max_abs':0.0,'worst_tensors':[]}
 def walk(x,y,key):
  if isinstance(x,torch.Tensor):
   row['tensors_checked']+=1
   if not isinstance(y,torch.Tensor) or x.shape!=y.shape or x.dtype!=y.dtype:row['structure_differences']+=1;return
   if x.stride()!=y.stride():row['stride_differences']+=1
   if torch.equal(x,y):return
   row['different_tensors']+=1
   if not x.is_floating_point():return
   maximum=0.0;diff2=0.0;reference2=0.0;count=0
   xx=x.reshape(-1);yy=y.reshape(-1)
   for offset in range(0,xx.numel(),1024*1024):
    old=xx[offset:offset+1024*1024].double();new=yy[offset:offset+1024*1024].double();assert torch.isfinite(old).all() and torch.isfinite(new).all()
    diff=(new-old).abs();maximum=max(maximum,float(diff.max()) if diff.numel() else 0.0);diff2+=float((diff*diff).sum());reference2+=float((old*old).sum());count+=int(torch.count_nonzero(diff))
   row['max_abs']=max(row['max_abs'],maximum);row['worst_tensors'].append({'path':key,'max_abs':maximum,'relative_L2':math.sqrt(diff2/reference2) if reference2 else None,'different_elements':count,'elements':x.numel()});row['worst_tensors']=sorted(row['worst_tensors'],key=lambda z:z['max_abs'],reverse=True)[:3]
  elif isinstance(x,dict):
   if not isinstance(y,dict) or x.keys()!=y.keys():row['structure_differences']+=1;return
   for name in x:walk(x[name],y[name],key+'.'+str(name))
  elif isinstance(x,(list,tuple)):
   if type(x)!=type(y) or len(x)!=len(y):row['structure_differences']+=1;return
   for i,(u,v) in enumerate(zip(x,y)):walk(u,v,key+'['+str(i)+']')
  elif type(x)!=type(y) or x!=y:row['different_primitive_values']+=1
 walk(a,b,'root');row['semantic_equal']=row['different_tensors']==row['different_primitive_values']==row['structure_differences']==0;rows.append(row);del a,b
assert not torch.cuda.is_initialized()
value={'state':'NATIVE_OLD_NEW_DIFFERENT_BYTE_ARTIFACTS_VALUE_COMPARISON_COMPLETE','pair_index_SHA256':sha(ROOT/'PAIR_INDEX.json'),'total_original_artifacts':200,'identical_full_bytes_artifacts':112,'different_bytes_artifacts':len(rows),'serialization_only_artifacts':sum(x['semantic_equal'] for x in rows),'numerically_or_structurally_different_artifacts':sum(not x['semantic_equal'] for x in rows),'rows':rows,'wall_seconds':time.monotonic()-started,'GPU_initialized':False,'acceptance_threshold_supplied':False,'model_qualified':False,'cross_platform_qualified':False,'note':'No tolerance is invented. Differences between separate native runs do not alone identify the cause as observer behavior or kernel/reduction nondeterminism.'}
with (ROOT/'result.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps({k:v for k,v in value.items() if k!='rows'}))
