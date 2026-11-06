"""Frozen full16owner Model-entry resources and non-executing source verification."""
from pathlib import Path
import hashlib,json,os,time
D=Path(__file__).resolve().parent;HOST=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');R=Path('/refactor')if str(D).startswith('/refactor/')else HOST
NAME='GLM-C1-layer-observer-r474/full-model-screen32-v1';LABEL='GLM-global-TP16-C1-observer-screen32-r474-v1';OUT=R/'results'/LABEL;NODES=['a02r3n15','a02r3n16'];IPS=['10.1.2.55','10.1.2.56'];PORT=43844;GPU_SECONDS=18000;WORK_SECONDS=17400;KEEPER_SECONDS=18600
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
 assert H(D/'GPU_MANIFEST.json')==pin;m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert m['GPU_GO']and m['Model_execution_authorized']and not m['Model_qualified'];assert H(D/'SOURCE_MANIFEST.json')==m['source_manifest_SHA256'];s=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==h for n,h in s['files_SHA256'].items());assert H(D/'GPU_RESOURCE_CLAIM.json')==m['claim_SHA256'];bind=json.loads((D/'CPU_BINDING.json').read_bytes());p=actual(bind['receipt']);assert H(p)==bind['receipt_SHA256'];cpu=json.loads(p.read_bytes());assert cpu['completed']and cpu['passed']and cpu['actual_CPU']==bind['actual_CPU']=='910407.1119'and cpu['source_manifest_SHA256']==bind['CPU_source_manifest_SHA256']and cpu['claim_SHA256']==bind['CPU_claim_SHA256']and cpu['all5_MD_equal_fresh_M']and cpu['all5_fresh_M_before_compile']and cpu['new_HOST']==5 and cpu['new_GPU']==0;assert cpu['library']==bind['library']and cpu['library_SHA256']==bind['library_SHA256']and len(cpu['all32_linked_objects_SHA256'])==32 and cpu['strong_parent_exports_retained']==273 and len(cpu['new_transitive_library_SHA256'])==71;assert H(actual(cpu['library']))==cpu['library_SHA256'];assert H(D/'observer_trial.py')==bind['copied_observer_trial_SHA256'];assert H(D/'typed_binding.py')==bind['copied_typed_binding_SHA256']
 for p,h in bind['actual_project_DSO_SHA256'].items():assert H(actual(p))==h,p
 inp=json.loads((D/'INPUT_BINDING.json').read_bytes());trial=json.loads((D/'TRIAL.json').read_bytes());assert trial['planned']==33 and trial['schedule']==__import__('resident_trial').schedule()and trial['phases']==__import__('resident_trial').phases();plan=actual(inp['private_plan']['path']);assert H(plan)==inp['private_plan']['SHA256'];v=json.loads(plan.read_bytes());assert v['planned']==32 and[rr['ordinal']for rr in v['requests']]==inp['selected_ordinals']and len({rr['case_id']for rr in v['requests']})==32 and sum(rr['suite']=='gsm8k'for rr in v['requests'])==16 and sum(rr['suite']=='mmlu'for rr in v['requests'])==16 and all(rr['max_new_tokens']==4096 and rr['prefill_window']==4096 for rr in v['requests']);
 for rr in v['requests']:
  meta=next(a for a in inp['private_plan']['requests']if a['ordinal']==rr['ordinal']);assert all(rr[k]==meta[k]for k in ['case_id','suite','prompt_tokens','prompt_IDs_SHA256','original_private_SHA256','body_SHA256','max_new_tokens','prefill_window']);assert rr['body'].get('temperature')==0.0 and len(rr['prompt_token_ids'])+4096<=32768
  assert hashlib.sha256(json.dumps(rr['prompt_token_ids'],separators=(',',':')).encode()).hexdigest()==rr['prompt_IDs_SHA256']
 model=actual(inp['model_path']);assert H(model/'config.json')==inp['model_config_SHA256'];assert H(model/'model.safetensors.index.json')==inp['model_index_SHA256'];assert H(actual(inp['tokenizer']['path']))==inp['tokenizer']['SHA256'];return m,cpu,inp,v
