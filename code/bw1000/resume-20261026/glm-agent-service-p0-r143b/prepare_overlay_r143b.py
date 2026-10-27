"""CPU static preparation only; no native/Transformer/Torch imports."""
import ast,datetime,hashlib,importlib.util,json,os,pathlib,sys,traceback
R=pathlib.Path('/refactor');D=R/'incoming/GLM-agent-service-r143b';B=R/'candidates/GLM-agent-service-r143b';BASE=R/'candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
q={'completed':False,'CPU_only':True,'static_only':True,'GPU_initialized':False,'native_model_constructed':False,'standard_server_imported':False,'protocol_commit':'7ceb60d656074376b911c2a232196a9457dfa6a6','actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
B.mkdir(exist_ok=False)
try:
 m=json.loads((D/'source-manifest-r143b.json').read_bytes());q['source_manifest_SHA256']=sha(D/'source-manifest-r143b.json')
 for name,h in m['source_SHA256'].items():
  p=D/name;assert sha(p)==h,name
  if p.suffix=='.py':ast.parse(p.read_text())
 parent=R/'candidates/GLM-critical-observer-r131/CPU-build.json';assert sha(parent)=='8c478b86c887d8f235c3eaa8ee9cd369d053112bc5fb437eb5bd0a8bb9f4a88d';cp=json.loads(parent.read_bytes());assert cp['completed']and cp['library_SHA256']=='ce830c4d77fe193e4a6e82597aa4fb5b50019c82edce542eb6d61217a238bf68'and sha(cp['library'])==cp['library_SHA256']
 for name in ['fastapi','uvicorn','jsonschema','transformers','tokenizers','numpy','safetensors']:
  assert importlib.util.find_spec(name)is not None,('missing_dependency',name)
 dst=B/'python/infinilm';src=BASE/'python/infinilm';assert src.is_dir();dst.mkdir(parents=True);basefiles={}
 for p in sorted(src.rglob('*')):
  rel=p.relative_to(src)
  if '__pycache__'in rel.parts or p.suffix=='.pyc':continue
  target=dst/rel
  if p.is_dir():target.mkdir(exist_ok=True)
  elif p.is_file():
   target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(p)
   if p.suffix=='.py'or '.so'in p.name:basefiles[str(p)]=sha(p)
 overlays={}
 for rel,h in m['overlay_SHA256'].items():
  source=D/'overlay'/rel;assert sha(source)==h
  target=B/rel;target.parent.mkdir(parents=True,exist_ok=True)
  if target.is_symlink():target.unlink()
  assert not target.exists();target.write_bytes(source.read_bytes());overlays[str(target)]=h
 model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');cfg=json.loads((model/'config.json').read_bytes());assert cfg['num_hidden_layers']==78
 q.update(completed=True,normal_CPU_exit=True,model_path=str(model),model_config_SHA256=sha(model/'config.json'),context_capacity=640,rows=32,TP=8,PP=2,cards=16,private_overlay=str(B/'python'),baseline_files_SHA256=basefiles,overlay_files_SHA256=overlays,Native_DSO=cp['library'],Native_DSO_SHA256=cp['library_SHA256'],runtime_import_qualification='GPU_PREWEIGHT_IMPORT_REQUIRED',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
(B/'CPU-prepare.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','actual_step','CPU_only','static_only','error']}));assert q['completed']
