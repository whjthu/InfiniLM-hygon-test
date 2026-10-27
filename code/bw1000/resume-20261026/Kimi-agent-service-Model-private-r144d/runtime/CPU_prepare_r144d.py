"""Pure static overlay preparation; never import native/Torch/Transformers."""
import pathlib,json,hashlib,ast,os,datetime,sys
R=pathlib.Path('/refactor');D=R/'incoming/Kimi-agent-service-Model-r144d';B=R/'candidates/Kimi-agent-service-r144d';BASE=R/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert not B.exists();B.mkdir();q={'completed':False,'CPU_only':True,'GPU':False,'native_imported':False,'Torch_imported':False,'model_constructed':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':os.environ.get('SLURMD_NODENAME',''),'baseline':str(BASE)}
try:
 m=json.loads((D/'service-source-manifest-r144d.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 for p in D.rglob('*.py'):ast.parse(p.read_bytes())
 count_path=R/'results/Kimi-agent-token-count-r143.json'
 assert sha(count_path)==m['required_root_token_count_receipt_SHA256']
 count=json.loads(count_path.read_bytes())
 assert count['CPU_only'] is True and count['GPU'] is False and count['weights_read'] is False
 assert count['all_fit_512'] is True and count['all_fit_1024_plus128'] is True
 assert all(x['truncated'] is False for x in count['results'])
 tokenizer=count['tokenizer'];model=pathlib.Path('/shared/Kimi-K3')
 assert sha(model/'tiktoken.model')==tokenizer['tiktoken_model_SHA256']
 assert sha(model/'tokenizer_config.json')==tokenizer['tokenizer_config_SHA256']
 q['simulated_budget_receipt_SHA256']=sha(count_path)
 q['simulated_budget_max_prompt_tokens']=max(x['prompt_tokens'] for x in count['results'])
 q['actual_tokenizer_pins']=tokenizer
 q['simulated_budget_is_model_qualification']=False
 src=BASE/'python/infinilm';dst=B/'python/infinilm';assert src.is_dir();dst.mkdir(parents=True);basepins={}
 for p in sorted(src.rglob('*')):
  rel=p.relative_to(src)
  if '__pycache__'in rel.parts or p.suffix=='.pyc':continue
  target=dst/rel
  if p.is_symlink() and p.is_dir():target.symlink_to(p.resolve(),target_is_directory=True)
  elif p.is_dir():target.mkdir(exist_ok=True)
  elif p.is_file():
   target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(p)
   if p.suffix=='.py'or '.so'in p.name:basepins[str(p)]=sha(p)
 overlays={}
 for rel,h in m['overlay_SHA256'].items():
  p=D/'overlay'/rel;assert sha(p)==h;target=B/rel;target.parent.mkdir(parents=True,exist_ok=True)
  assert target.parent.resolve().is_relative_to(B.resolve()),'overlay parent escapes private candidate'
  if target.is_symlink():target.unlink()
  assert not target.exists();target.write_bytes(p.read_bytes());overlays[str(target)]=sha(target)
 assert 'torch'not in sys.modules and 'transformers'not in sys.modules
 q.update(completed=True,source_manifest_SHA256=sha(D/'service-source-manifest-r144d.json'),baseline_files_SHA256=basepins,overlay_files_SHA256=overlays,private_python=str(B/'python'),standard_server_imported=False,model_or_HTTP_qualification=False,normal_CPU_exit=True)
except BaseException as e:q['error']=repr(e)
(B/'CPU-overlay-prepared.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','actual_step','node','error']}));assert q['completed']
