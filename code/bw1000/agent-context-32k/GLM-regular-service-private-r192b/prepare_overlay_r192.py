"""CPU static preparation only; no native/Transformer/Torch imports."""
import ast,datetime,hashlib,importlib.util,json,os,pathlib,sys,traceback
R=pathlib.Path('/refactor');D=R/'incoming/GLM-regular-service-r192';B=R/'candidates/GLM-regular-service-r192';BASE=R/'candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
q={'completed':False,'CPU_only':True,'static_only':True,'GPU_initialized':False,'native_model_constructed':False,'standard_server_imported':False,'protocol_commit':'7ceb60d656074376b911c2a232196a9457dfa6a6 + scoped-timeout-correction + GLM namedtools preserve-all patch; agent_protocol_SHA256=c91ab84a5ddc800602dcae0319ad5c02ad3b01c1936a2bd57bce90c2ad15abc3','actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
B.mkdir(exist_ok=False)
try:
 m=json.loads((D/'source-manifest-r192.json').read_bytes());q['source_manifest_SHA256']=sha(D/'source-manifest-r192.json')
 for name,h in m['source_SHA256'].items():
  p=D/name;assert sha(p)==h,name
  if p.suffix=='.py':ast.parse(p.read_text())
 from allocation_proof_r192 import verify as allocation_verify
 q['allocation795_proof']=allocation_verify()
 parent=R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json';assert sha(parent)=='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529';cp=json.loads(parent.read_bytes());assert cp['completed']and cp['library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'and sha(cp['library'])==cp['library_SHA256']
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
 model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');cfg=json.loads((model/'config.json').read_bytes());assert cfg['num_hidden_layers']==78 and cfg['max_position_embeddings']>=32768
 child=dict(os.environ,USE_TORCH='0',USE_TF='0');z=__import__('subprocess').run([sys.executable,'-B',str(D/'CPU_contract_r192.py')],env=child,capture_output=True,text=True,timeout=180);assert z.returncode==0,(z.stdout,z.stderr);q['protocol_and_budget_CPU_contract']=json.loads(z.stdout)
 wire=__import__('subprocess').run([sys.executable,'-B',str(D/'CPU_wire_contract_r178a.py')],env=child,capture_output=True,text=True,timeout=30);assert wire.returncode==0,(wire.stdout,wire.stderr);q['actual_WireClient_budget_CPU_contract']=json.loads(wire.stdout)
 q.update(completed=True,normal_CPU_exit=True,model_path=str(model),model_config_SHA256=sha(model/'config.json'),context_capacity=32768,rows=1,TP=8,PP=4,cards=32,private_overlay=str(B/'python'),baseline_files_SHA256=basefiles,overlay_files_SHA256=overlays,Native_DSO=cp['library'],Native_DSO_SHA256=cp['library_SHA256'],runtime_import_qualification='GPU_PREWEIGHT_IMPORT_REQUIRED',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
(B/'CPU-prepare.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','actual_step','CPU_only','static_only','error']}));assert q['completed']
