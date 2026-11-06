"""Lossless integer path-index encoding of an already validated local snapshot.
No new original reads, preprocessing, SDK, compiler, or device work.
"""
from pathlib import Path
import argparse,hashlib,json
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def compact(q):
 keys=sorted(q['path_SHA256']);index={p:i for i,p in enumerate(keys)}
 out={k:v for k,v in q.items()if k!='units'};out['schema']=2;out['path_index_order']='sorted(path_SHA256)';out['units']={}
 for key,u in q['units'].items():
  r={k:v for k,v in u.items()if k!='dependency_paths'};r['dependency_path_indices']=[index[p]for p in u['dependency_paths']]
  assert [keys[i]for i in r['dependency_path_indices']]==u['dependency_paths']
  out['units'][key]=r
 assert len(out['units'])==50 and len(keys)==1755
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args();q=json.loads(a.input.read_bytes());r=compact(q);r['full_snapshot_original']={'private':'L/'+a.input.name,'SHA256':H(a.input)};r['index_encoder_SHA256']=H(__file__);a.output.write_text(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8');print(json.dumps({'units':len(r['units']),'paths':len(r['path_SHA256']),'lossless_on_all50':True,'bytes':a.output.stat().st_size,'SHA256':H(a.output)}))
