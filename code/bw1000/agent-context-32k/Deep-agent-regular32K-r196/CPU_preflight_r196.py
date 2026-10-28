"""Actual new-artifact/data preparation. No Torch/Native/weights/GPU."""
import pathlib,json,hashlib,os,sys,ast,time,traceback
from native_binding_r196 import binding,BASE
from request_contract_r196 import CPU_contract
from profiles_r196 import prepare_actual
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');B=R/'candidates/Deep-agent-regular32K-r196';MODEL=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');CORE=R/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();start=time.monotonic()
assert not B.exists();B.mkdir();q={'completed':False,'CPU_only':True,'Native_loaded':False,'GPU_initialized':False,'weights_payload_opened':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 m=json.loads((D/'manifest-r196.json').read_bytes())
 for p,h in m['source_SHA256'].items():assert sha(D/p)==h,p
 for p in D.rglob('*.py'):ast.parse(p.read_bytes())
 b,c,n=binding();q.update(Native_binding_SHA256=sha(D/'Native-binding.json'),new_Native_path=c['library'],new_Native_SHA256=c['library_SHA256'],new_Native_CPU_receipt_SHA256=b['CPU_receipt_SHA256']);q['budget_contract']=CPU_contract()
 assert sha(BASE)=='d206637dff3e79d17756962974ed6935e85299d137f602b988d8688d29d999d4'
 from bootstrap_r196 import transform
 _,q['loader_AST_contract']=transform(BASE.read_bytes())
 from prompt_contract_r196 import run as prompt_contract
 q['saved_request_prompt_contract']=prompt_contract(MODEL);q['profiles']=prepare_actual(MODEL,B/'profiles')
 # Private package with byte-pinned public protocol overlay. No package import.
 src=CORE/'python/infinilm';dst=B/'python/infinilm';dst.mkdir(parents=True);basepins={}
 for p in src.rglob('*'):
  rel=p.relative_to(src)
  if '__pycache__'in rel.parts or p.suffix=='.pyc':continue
  target=dst/rel
  if p.is_dir():target.mkdir(exist_ok=True)
  elif p.is_file():
   target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(p)
   if p.suffix=='.py'or'.so'in p.name:basepins[str(p)]=sha(p)
 overlays={}
 for rel,h in m['overlay_SHA256'].items():
  p=D/'overlay'/rel;assert sha(p)==h;target=B/rel;target.parent.mkdir(parents=True,exist_ok=True)
  if target.is_symlink():target.unlink()
  assert not target.exists();target.write_bytes(p.read_bytes());overlays[str(target)]=sha(target)
 assert 'torch'not in sys.modules and 'transformers'not in sys.modules and '_infinilm'not in sys.modules
 q.update(completed=True,model_directory=str(MODEL),source_manifest_SHA256=sha(D/'manifest-r196.json'),baseline_files_SHA256=basepins,overlay_files_SHA256=overlays,scope='Actual CPU artifact/data/official-Rust profiles and source/budget contracts only; new NativeGPU/create/CCL/load/long agent/quality NOT_RUN')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q.update(Torch_imported='torch'in sys.modules,Transformers_imported='transformers'in sys.modules,Native_imported='_infinilm'in sys.modules,seconds=time.monotonic()-start);(B/'CPU-preflight.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','actual_step','new_Native_SHA256','Torch_imported','error','seconds']}));assert q['completed']
