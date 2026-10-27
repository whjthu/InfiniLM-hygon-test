"""Static checkpoint metadata/AST/dependency discovery and private overlay only.
No model construction, weights/tensor data, GPU initialization or network access.
"""
import pathlib,json,hashlib,os,sys,datetime,importlib.util,importlib.metadata,struct,traceback,ast
R=pathlib.Path('/refactor');D=R/'incoming/toolcall-validation-CPU-r138';B=R/'candidates/toolcall-Qwen06-r138';BASE=R/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM';MODEL=pathlib.Path('/shared/Qwen3-0.6B');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert all(os.environ.get(n)=='-1'for n in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
assert not B.exists();B.mkdir();q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'CPU_only':True,'GPU_initialized':False,'weights_tensor_values_read':False,'native_model_constructed':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'baseline':str(BASE),'model':str(MODEL),'protocol_commit':'7ceb60d656074376b911c2a232196a9457dfa6a6'}
try:
 manifest=D/'CPU-source-manifest.json';m=json.loads(manifest.read_bytes());q['source_manifest_SHA256']=sha(manifest)
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 c=MODEL/'config.json';tc=MODEL/'tokenizer_config.json';assert sha(c)=='660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd'and sha(tc)=='d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'
 cfg=json.loads(c.read_bytes());assert cfg['model_type']=='qwen3'and cfg['hidden_size']==1024 and cfg['num_hidden_layers']==28
 weight=MODEL/'model.safetensors';assert weight.stat().st_size==1503300328
 with weight.open('rb')as f:
  header_length=struct.unpack('<Q',f.read(8))[0];assert 0<header_length<16*1024**2;header_bytes=f.read(header_length);header=json.loads(header_bytes)
 tensors={k:v for k,v in header.items()if k!='__metadata__'};assert len(tensors)>200
 for n,t in tensors.items():
  lo,hi=t['data_offsets'];assert type(lo)is int and type(hi)is int and 0<=lo<=hi<=weight.stat().st_size-header_length-8
 q['checkpoint_metadata']={'config_SHA256':sha(c),'tokenizer_config_SHA256':sha(tc),'safetensors_bytes':weight.stat().st_size,'header_length':header_length,'header_SHA256':hashlib.sha256(header_bytes).hexdigest(),'tensor_count':len(tensors),'dtypes':sorted({x['dtype']for x in tensors.values()}),'tensor_data_read':False}
 required=['numpy','safetensors','tokenizers','transformers','jinja2','janus','xxhash','fastapi','uvicorn','jsonschema']
 q['dependencies']={}
 for n in required:
  try:version=importlib.metadata.version(n)
  except importlib.metadata.PackageNotFoundError:version=None
  q['dependencies'][n]={'exists':importlib.util.find_spec(n)is not None,'version':version}
 assert all(x['exists']for x in q['dependencies'].values()),('missing_dependency',q['dependencies'])
 # Preserve all current native libraries. Ordinary Python files are private symlinks;
 # only nine audited protocol files are copied into this isolated overlay.
 dst=B/'python/infinilm';src=BASE/'python/infinilm';assert src.is_dir();dst.mkdir(parents=True)
 baseline_files={}
 for p in sorted(src.rglob('*')):
  rel=p.relative_to(src)
  if '__pycache__'in rel.parts or p.suffix=='.pyc':continue
  target=dst/rel
  if p.is_dir():target.mkdir(exist_ok=True)
  elif p.is_file():
   target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(p)
   if p.suffix=='.py'or '.so'in p.name:baseline_files[str(p)]=sha(p)
 overlays={}
 for rel,h in m['overlay_SHA256'].items():
  p=D/'overlay'/rel;assert sha(p)==h
  target=B/rel;target.parent.mkdir(parents=True,exist_ok=True)
  if target.is_symlink():target.unlink()
  assert not target.exists();target.write_bytes(p.read_bytes());overlays[str(target)]=sha(target)
 q['baseline_files_SHA256']=baseline_files;q['overlay_files_SHA256']=overlays
 # Do not import Torch, Transformers, tokenizer, InfiniLM, server or native code.
 # find_spec above uses top-level module names only and does not execute them.
 for rel in m['overlay_SHA256']:
  p=B/rel;ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
 template=json.loads(tc.read_bytes()).get('chat_template');template_text=json.dumps(template,ensure_ascii=False)
 assert template and 'tools'in template_text and '<tool_call>'in template_text
 q['template_static']={'present':True,'tools_signal':True,'tool_call_signal':True,'tokenized':False,'rendered':False}
 q['GPU_initialized']=False
 q['packages_imported_by_this_preflight']=['stdlib_only']
 q['runtime_import_qualification']='DEFERRED_TO_ONE_CARD_GPU_PREWEIGHT_SMOKE'
 q['standard_server_imported']=False;q['native_modules']={}
 q.update(completed=True,static_only=True,static_overlay_AST_passed=True,model_qualification=False,HTTP_qualification=False,private_overlay=str(B/'python'),normal_CPU_exit=True)

except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
(B/'CPU-preflight.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','actual_step','static_only','GPU_initialized','error']}));assert q['completed']
