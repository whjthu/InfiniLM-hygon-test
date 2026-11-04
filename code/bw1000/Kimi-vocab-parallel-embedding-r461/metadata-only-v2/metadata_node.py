"""Phase-marked scalar metadata only. No Torch/SDK/Native/Slurm execution."""
import os,time,sys
os.write(2,('META_PHASE python.entry unix=%.9f pid=%d\n'%(time.time(),os.getpid())).encode())
import json,hashlib,struct
from pathlib import Path
def phase(name,**details):
 os.write(2,(json.dumps(dict(phase=name,unix=time.time(),monotonic=time.monotonic(),PID=os.getpid(),**details),separators=(',',':'))+'\n').encode())
P=Path('/refactor/incoming/Kimi-vocab-parallel-embedding-r461/device-v1')
assert os.environ.get('K461_METADATA_ONLY')=='1'
phase('parent.import.before')
sys.path.insert(0,str(P))
from constructor import check_metadata
import embedding
phase('parent.import.return')
def sha(p):
 phase('file_hash.before',path=str(p))
 b=Path(p).read_bytes();value=hashlib.sha256(b).hexdigest()
 phase('file_hash.return',path=str(p),bytes=len(b),SHA256=value)
 return value
def header(q):
 pin=q['checkpoint_embedding'];p=Path(q['model_directory'])/pin['filename']
 phase('header.open.before',path=str(p))
 with p.open('rb')as f:
  n=struct.unpack('<Q',f.read(8))[0];assert n==pin['header_bytes'];raw=f.read(n)
 phase('header.read.return',bytes=n)
 assert hashlib.sha256(raw).hexdigest()==pin['header_SHA256']
 phase('header.json.before');value=json.loads(raw)[embedding.KEY];phase('header.json.return')
 assert value==pin['tensor']
 phase('header.stat.before');st=p.stat();phase('header.stat.return',bytes=st.st_size,mtime_ns=st.st_mtime_ns)
 assert st.st_size==pin['file_bytes']and st.st_mtime_ns==pin['file_mtime_ns']
 return dict(header_SHA256=pin['header_SHA256'],tensor=value)
def main():
 phase('package_config.read.before');q=json.loads((P/'config.json').read_bytes());phase('package_config.read.return')
 roles=[]
 for n,h in enumerate(q['nodes']):
  phase('role.begin',node=n,recording_hostname=h)
  assert n in range(4)
  for key,value in q['environment'].items():assert os.environ.get(key)==value,(key,os.environ.get(key))
  root=Path(q['model_directory'])
  assert sha(root/'config.json')==q['config_SHA256']
  assert sha(root/'model.safetensors.index.json')==q['index_SHA256']
  phase('model_config.json.before');cfg=json.loads((root/'config.json').read_bytes());phase('model_config.json.return')
  phase('model_index.json.before');wm=json.loads((root/'model.safetensors.index.json').read_bytes())['weight_map'];phase('model_index.json.return',keys=len(wm))
  phase('check_metadata.before');check_metadata(cfg,wm);phase('check_metadata.return')
  assert wm[embedding.KEY]==q['checkpoint_embedding']['filename'];header(q)
  assert[len(v)for v in embedding.queries()]==q['lookup_widths']==[1,1,64,512]
  roles.append(dict(node=n,hostname_leaf='recording:'+h));phase('role.return',node=n)
 phase('result.emit.before')
 final_header=header(q) # preserve the original final metadata-result header read
 print(json.dumps(dict(passed=True,roles=roles,active_candidate_Native_calls=0,explicit_SDK_calls=0,
  original_preload_mapping_or_initialization='not asserted; original environment retained',GPU=0,Slurm=0,
  real_affinity_count=len(os.sched_getaffinity(0)),GPU_entry_mask_qualified=False,metadata_only=True,
  query_ID_SHA256=[embedding.ID_SHA(v)for v in embedding.queries()],checkpoint_header=final_header)),flush=True)
 phase('result.emit.return')
if __name__=='__main__':main()
