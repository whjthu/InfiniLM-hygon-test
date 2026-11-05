"""Frozen full16owner Model-entry resources and non-executing source verification."""
from pathlib import Path
import hashlib,json,os,time
D=Path(__file__).resolve().parent;HOST=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');R=Path('/refactor')if str(D).startswith('/refactor/')else HOST
NAME='GLM952-global-TP16-component-r470/full-model-entry-v1';LABEL='GLM-global-TP16-full-Model-entry-r470-v1';OUT=R/'results'/LABEL;NODES=['a02r3n15','a02r3n16'];IPS=['10.1.2.55','10.1.2.56'];PORT=43844;GPU_SECONDS=14400;WORK_SECONDS=13800;KEEPER_SECONDS=15000
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def actual(name):
 s=str(name)
 if s.startswith('/refactor/'):return R/s[len('/refactor/'):]
 if s.startswith('/old/')and R==HOST:return Path('/public/home/aclsylqidf/hygon-inference-20261013')/s[len('/old/'):]
 if s.startswith(str(HOST)+'/')and R!=HOST:return R/s[len(str(HOST)+'/'):]
 return Path(s)
def save(p,q):
 p=Path(p);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()));tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def verify(pin):
 assert H(D/'GPU_MANIFEST.json')==pin;m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert m['GPU_GO']and m['Model_execution_authorized']and not m['Model_qualified'];assert H(D/'SOURCE_MANIFEST.json')==m['source_manifest_SHA256'];s=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==h for n,h in s['files_SHA256'].items());assert H(D/'GPU_RESOURCE_CLAIM.json')==m['claim_SHA256'];bind=json.loads((D/'CPU_BINDING.json').read_bytes());p=actual(bind['receipt']);assert H(p)==bind['receipt_SHA256'];cpu=json.loads(p.read_bytes());assert cpu['passed']and cpu['actual_CPU']=='910407.1087'and cpu['source_manifest_SHA256']==bind['CPU_source_manifest_SHA256']and cpu['parent_source_manifest_SHA256']==bind['production_source_manifest_SHA256'];assert H(actual(cpu['library']))==bind['library_SHA256']==cpu['library_SHA256'];assert H(D/'typed_binding.py')==bind['copied_typed_binding_SHA256']
 for p,h in bind['actual_project_DSO_SHA256'].items():assert H(actual(p))==h,p
 inp=json.loads((D/'INPUT_BINDING.json').read_bytes());plan=actual(inp['private_plan']['path']);assert H(plan)==inp['private_plan']['SHA256'];v=json.loads(plan.read_bytes());assert v['planned']==2 and[rr['ordinal']for rr in v['requests']]==[0,128]and all(rr['max_new_tokens']==4096 and rr['prefill_window']==4096 for rr in v['requests']);
 for rr in v['requests']:
  assert rr['body'].get('temperature')==0.0 and len(rr['prompt_token_ids'])+4096<=32768
  assert hashlib.sha256(json.dumps(rr['prompt_token_ids'],separators=(',',':')).encode()).hexdigest()==rr['prompt_IDs_SHA256']
 model=actual(inp['model_path']);assert H(model/'config.json')==inp['model_config_SHA256'];assert H(model/'model.safetensors.index.json')==inp['model_index_SHA256'];assert H(actual(inp['tokenizer']['path']))==inp['tokenizer']['SHA256'];return m,cpu,inp,v
