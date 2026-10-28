"""Reuse actual CPU568; only relocate private data/package and verify admission fix."""
import pathlib,json,hashlib,os,sys,ast,time,traceback,shutil
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');B=R/'candidates/Deep-agent-regular32K-r206b';P=R/'candidates/Deep-agent-regular32K-r206';S=R/'incoming/Deep-agent-regular32K-r206';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();start=time.monotonic()
assert not B.exists();B.mkdir();q={'completed':False,'CPU_only':True,'Native_loaded':False,'GPU_initialized':False,'weights_payload_opened':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 m=json.loads((D/'manifest-r206b.json').read_bytes())
 for p,h in m['source_SHA256'].items():assert sha(D/p)==h,p
 for p in D.rglob('*.py'):ast.parse(p.read_bytes())
 assert sha(S/'manifest-r206.json')==m['parent_r206_manifest_SHA256']=='54a285759c2a3e662c8699cdd4d7fef9737791b61395f21419c4b0cbd8365aaf'
 parent=json.loads((S/'manifest-r206.json').read_bytes())
 for p,h in parent['source_SHA256'].items():assert sha(S/p)==h,p
 for new,old in m['parent_equivalent_source'].items():
  value=(D/new).read_bytes().replace(b'\r\n',b'\n').replace(b'r206b',b'r206').replace(b'R206B',b'R206')
  assert value==(S/old).read_bytes().replace(b'\r\n',b'\n'),new
 oldcpu=P/'CPU-preflight.json';assert sha(oldcpu)==m['reused_actual_CPU568']['SHA256']=='ee34e72b6ad407d909e0bd5ba3d169acb056a40aed6391ed78899b7a290de271'
 old=json.loads(oldcpu.read_bytes());assert old['completed']and old['CPU_only']and old['actual_step']=='910407.568'
 assert not any(old[k]for k in ['Torch_imported','Transformers_imported','Native_imported','Native_loaded','GPU_initialized','weights_payload_opened'])
 assert old['source_manifest_SHA256']==m['parent_r206_manifest_SHA256']and old['Native_binding_SHA256']==sha(D/'Native-binding.json')
 assert old['budget_contract']['passed']and old['saved_request_prompt_contract']['passed']and old['history_contribution_CPU_contract']['passed']and old['actual_saved_history_contribution_contract']['passed']
 assert len(old['history_contribution_CPU_contract']['cases'])==7 and len(old['actual_saved_history_contribution_contract']['cases'])==4
 from CPU_admission_contract_r206b import run
 q['admission_CPU_contract']=run();assert q['admission_CPU_contract']['passed']
 # Reuse actual Rust-prepared data. Runtime still checks real HF/Rust fixture IDs.
 for p,h in old['baseline_files_SHA256'].items():assert sha(p)==h,p
 for p,h in old['overlay_files_SHA256'].items():assert sha(p)==h,p
 shutil.copytree(P/'python',B/'python',symlinks=True)
 shutil.copytree(P/'profiles',B/'profiles',symlinks=False)
 overlays={}
 for rel,h in m['overlay_SHA256'].items():
  target=B/rel;assert sha(target)==h==sha(D/'overlay'/rel);overlays[str(target)]=h
 # Input_file retains immutable parent CPU568 paths. Actual per-run client work
 # directories remain new and isolated; no generated assistant turn is copied.
 assert (P/'profiles/profiles.json').read_bytes()==(B/'profiles/profiles.json').read_bytes()
 for profile in json.loads((B/'profiles/profiles.json').read_bytes()):assert sha(profile['input_file'])==profile['input_SHA256']
 for key in ['approved_history_encoder_SHA256','Native_binding_SHA256','new_Native_path','new_Native_SHA256','new_Native_CPU_receipt_SHA256','budget_contract','loader_AST_contract','saved_request_prompt_contract','profiles','history_contribution_CPU_contract','actual_saved_history_contribution_contract','model_directory','baseline_files_SHA256']:q[key]=old[key]
 q.update(completed=True,source_manifest_SHA256=sha(D/'manifest-r206b.json'),overlay_files_SHA256=overlays,reused_actual_CPU568={'path':str(oldcpu),'SHA256':sha(oldcpu),'actual_step':old['actual_step'],'unrelated_contracts_rerun':False},profile_input_source='Immutable CPU568 profiles/input files; no model turns invented',scope='Only new private admission race/failure CPU contract plus relocation. Actual CPU568 renderer/data/budget/loader proof reused. No Native constructor, GPU, weights or Model rerun.')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q.update(Torch_imported='torch'in sys.modules,Transformers_imported='transformers'in sys.modules,Native_imported='_infinilm'in sys.modules,seconds=time.monotonic()-start)
assert not any(q[k]for k in ['Torch_imported','Transformers_imported','Native_imported'])
(B/'CPU-preflight.json').write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','actual_step','new_Native_SHA256','Torch_imported','error','seconds']}));assert q['completed']
