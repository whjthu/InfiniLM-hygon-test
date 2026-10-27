"""Read checkpoint metadata; make private Python overlay and import standard server.
No model construction, weights/tensor data, GPU initialization or network access.
"""
import pathlib,json,hashlib,os,sys,datetime,importlib.util,importlib.metadata,struct,traceback
R=pathlib.Path('/refactor');D=R/'incoming/toolcall-validation-r136';B=R/'candidates/toolcall-Qwen06-r136';BASE=R/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM';MODEL=pathlib.Path('/shared/Qwen3-0.6B');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
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
 sys.path[:0]=[str(B/'python'),str(BASE/'python')]
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
 tool={'type':'function','function':{'name':'calculate','description':'Add integers','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}
 messages=[{'role':'user','content':'Add two integers with calculate.'}]
 rendered=tok.apply_chat_template(messages,tools=[tool],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(rendered,add_special_tokens=False)
 assert '<tool_call>'in rendered and 'calculate'in rendered and 'tools'in rendered and len(ids)<2048
 q['template']={'token_count':len(ids),'rendered_SHA256':hashlib.sha256(rendered.encode()).hexdigest(),'hermes_tool_tag':True,'enable_thinking':False}
 import infinilm
 from infinilm.server.inference_server import InferenceServer
 from infinilm.server.tool_parsers import TOOL_CALL_FORMATS
 assert 'hermes'in TOOL_CALL_FORMATS
 q['server_module']=sys.modules[InferenceServer.__module__].__file__;assert pathlib.Path(q['server_module']).is_relative_to(B)
 q['native_modules']={n:{'path':v.__file__,'SHA256':sha(v.__file__)}for n,v in list(sys.modules.items())if (n.startswith('infinilm')or n=='_infinilm')and getattr(v,'__file__',None)and v.__file__.endswith('.so')}
 q['GPU_initialized']=bool('torch'in sys.modules and sys.modules['torch'].cuda.is_initialized());assert not q['GPU_initialized']
 q.update(completed=True,standard_server_imported=True,hermes_parser_imported=True,model_qualification=False,HTTP_qualification=False,private_overlay=str(B/'python'),normal_CPU_exit=True)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
(B/'CPU-preflight.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','actual_step','standard_server_imported','GPU_initialized','error']}));assert q['completed']
