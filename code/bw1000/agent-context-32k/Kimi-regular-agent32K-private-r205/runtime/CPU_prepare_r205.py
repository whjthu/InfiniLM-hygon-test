"""Pure static overlay preparation; never import native/Torch/Transformers."""
import pathlib,json,hashlib,ast,os,datetime,sys,subprocess
R=pathlib.Path('/refactor');D=R/'incoming/Kimi-agent-service-Model-r205';B=R/'candidates/Kimi-agent-service-r205';BASE=R/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'native_imported':False,'Torch_imported':False,'model_constructed':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':os.environ.get('SLURMD_NODENAME',''),'baseline':str(BASE)}
def verify_profiles():
 folder=R/'candidates/Kimi-long-agent32K-r179/regular-agent-profiles-r179'
 receipt=folder/'CPU-profile-receipt-r179.json'
 assert sha(receipt)=='d63db286e894e93fcd1eb2b5cf5893280f5b6ddae958ad41725153db78a4de4a'
 q=json.loads(receipt.read_bytes());assert q['CPU_only']and not q['GPU']and not q['weights_read']and not q['model_outputs_generated']
 assert q['official_ordinary_BPE_count']and q['max_context']==32768 and q['ordinary_default_output']==4096 and q['thinking_output']==8192
 profiles=json.loads((folder/'profiles.json').read_bytes());assert profiles==q['profiles']and len(profiles)==3
 assert [(x['label'],x['minimum_prompt_tokens'],x['output_budget'])for x in profiles]==[('HTTP',8192,4096),('SSE',16384,4096),('thinking',8192,8192)]
 for x in profiles:
  assert sha(pathlib.Path(x['input_file']))==x['input_SHA256']and x['minimum_prompt_tokens']<=x['CPU_raw_text_tokens']<x['minimum_prompt_tokens']+256
 assert sha(pathlib.Path('/shared/Kimi-K3/tiktoken.model'))=='b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103'
 assert sha(pathlib.Path('/shared/Kimi-K3/tokenizer_config.json'))=='5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e'
 return {'path':str(receipt),'SHA256':sha(receipt),'profiles_SHA256':sha(folder/'profiles.json'),'is_Model_qualification':False}
try:
 if B.exists():
  previous=json.loads((B/'CPU-overlay-prepared.json').read_bytes())
  assert previous['completed']and previous['CPU_only']and not previous['GPU']
  assert previous['source_manifest_SHA256']==sha(D/'service-source-manifest-r205.json')
  for path,h in previous['overlay_files_SHA256'].items():assert sha(path)==h,path
  for path,h in previous['baseline_files_SHA256'].items():assert sha(path)==h,path
  verify_profiles()
  print(json.dumps({'completed':True,'CPU_only':True,'GPU':False,'actual_step':q['actual_step'],'node':q['node'],'reused_exact_pinned_CPU_overlay':True,'source_manifest_SHA256':previous['source_manifest_SHA256']}));sys.exit(0)
 B.mkdir()
 m=json.loads((D/'service-source-manifest-r205.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 for p in D.rglob('*.py'):ast.parse(p.read_bytes())
 q['actual_profile_receipt']=verify_profiles()
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
 q.update(completed=True,source_manifest_SHA256=sha(D/'service-source-manifest-r205.json'),baseline_files_SHA256=basepins,overlay_files_SHA256=overlays,private_python=str(B/'python'),standard_server_imported=False,model_or_HTTP_qualification=False,normal_CPU_exit=True)
except SystemExit:raise
except BaseException as e:q['error']=repr(e)
(B/'CPU-overlay-prepared.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','actual_step','node','error']}));assert q['completed']
