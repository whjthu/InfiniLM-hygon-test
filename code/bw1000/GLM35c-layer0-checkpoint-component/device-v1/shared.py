"""Source/CPU/checkpoint metadata admission; no Torch/DSO import or execution."""
from pathlib import Path,PurePosixPath
import hashlib,json,os
from component_api import read_original_above
NAME='GLM35c-layer0-checkpoint-component/device-v1'
LABEL='GLM35c-layer0-checkpoint-device-v1'
ROOT_RESULT='GLM35c-layer0-checkpoint-device-v1'
GPU_SECONDS=600
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mapped(root,value):
 p=PurePosixPath(value)
 if not p.is_absolute()or ".."in p.parts:raise ValueError("Invalid metadata path")
 if p.is_relative_to('/refactor'):return Path(root).joinpath(*p.relative_to('/refactor').parts)
 if p.is_relative_to('/old'):return Path('/public/home/aclsylqidf/hygon-inference-20261013').joinpath(*p.relative_to('/old').parts)if str(root)!='/refactor'else Path(value)
 raise ValueError('foreign metadata domain: '+value)
def verify(root,expected_manifest=None):
 root=Path(root);d=root/'incoming'/NAME;mf=d/'GPU_SOURCE_MANIFEST.json'
 if expected_manifest is not None:assert sha(mf)==expected_manifest
 m=json.loads(mf.read_bytes());assert m['GPU_GO']is True and m['Model_GO']is False and m['component_only']is True
 for n,h in m['files_SHA256'].items():assert sha(d/n)==h,n
 raw=mapped(root,m['CPU_receipt']).read_bytes();assert hashlib.sha256(raw).hexdigest()==m['CPU_receipt_SHA256'];a=json.loads(raw)
 assert a['completed']and a['passed']and a['actual_CPU']=='910407.1054'and a['new_HOST']==1 and a['retained_objects']==29 and a['new_HIP']==0
 assert a['source_manifest_SHA256']==m['CPU_source_manifest_SHA256']and a['library_SHA256']==m['library_SHA256']and not a['Native_execution']and not a['dlopen']
 assert sha(mapped(root,a['library']))==m['library_SHA256']
 for p,h in a['objects_SHA256'].items():assert sha(mapped(root,p))==h,p
 for p,h in m['qualified_dependency_SHA256'].items():assert sha(mapped(root,p))==h,p
 model=mapped(root,m['model']);assert sha(model/'config.json')==m['config_SHA256']
 meta=json.loads((d/'CHECKPOINT_METADATA.json').read_bytes());assert sha(model/'model.safetensors.index.json')==meta['index_SHA256']
 assert sha(mapped(root,m['original_private_plan']))==m['original_plan_SHA256']
 above=read_original_above(mapped(root,m['original_private_plan']))
 assert m['resource']==dict(parent=910371,node='a02r3n03',cards=8,TP=8,PP=1,world=8,cores=32,memory_MiB=131072,GPU_seconds=600,owner_seconds=540,keeper_parent=910407,keeper_node='a14r2n20',keeper_cores=2,keeper_memory_MiB=8192,keeper_seconds=900)
 return m,a,above
