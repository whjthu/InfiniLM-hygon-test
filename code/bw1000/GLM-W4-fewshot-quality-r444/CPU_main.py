"""One bounded source/consumer plus actual checkpoint CPU encoding closure.

Native/model/device code is not run. Client HTTP/Model quality qualification is
explicitly outside this CPU receipt and still requires its own source closure.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,time,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/GLM-W4-fewshot-quality-r444-CPU-v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def main():
 started=time.monotonic();q=dict(passed=False,completed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=0,Model=0,Native_dlopen=False,Native_constructed=False,weights_loaded=False,compile=False,HTTP=0,whole_quality_client_main_executed=False,phases=[])
 try:
  assert not B.exists()and all(os.environ[k]=='-1'for k in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
  m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert sha(D/'SOURCE_MANIFEST.json')==os.environ['GLM_R444_CPU_SOURCE_SHA256']
  for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
  provider=Path('/refactor/incoming/public-GLM-native-provider-r425-ffi-v4');assert sha(provider/'SOURCE_MANIFEST.json')=='f5015c22a4434576a9b4cf00708dbe19594b70598a6b3e004410bfc63dbd9858'
  for n,h in json.loads((provider/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(provider/n)==h,n
  q['window_consumer']=load('r444_window_source_consumer',D/'check_window_audit_CPU.py').check(provider/'public_glm_native');assert q['window_consumer']['passed'];q['phases'].append('actual_original_execute_plan_with_recording_callbacks_and_reverse_source_patch')
  q['score_consumer']=load('r444_scoring_source_consumer',D/'check_scoring_CPU.py').check();assert q['score_consumer']['passed'];q['phases'].append('grade_and_planned_partial_denominator_contract')
  encoder=load('r444_actual_checkpoint_encoder',D/'CPU_encode.py');rc=encoder.main();q['phases'].append('actual_official_checkpoint_CPU_processor_all256')
  encoding=B/'encoding.json';value=json.loads(encoding.read_bytes());assert rc==0 and value['passed']and len(value['cases'])==256 and value['matching_body_pairs']==256 and value['actual_checkpoint_CPU_processor']and not value['Torch_CUDA_initialized']and not value['Native_constructed']and not value['weights_loaded']
  q.update(passed=True,source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),encoding_receipt=str(encoding),encoding_receipt_SHA256=sha(encoding),actual_unique_encodings=256,matching_body_pairs=256,all_full_fewshot_user_bodies_preserved=True,actual_live_window_counter_verified=False,planned_multigroup_cases=value['planned_multigroup_cases'],planned_multigroup_by_suite=value['planned_multigroup_by_suite'],prompt_tokens_min=value['prompt_tokens_min'],prompt_tokens_max=value['prompt_tokens_max'],whole_quality_client_main_executed=False)
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:
  B.mkdir(exist_ok=True);q.update(completed=True,elapsed_s=time.monotonic()-started)
  if Path('/proc/self/maps').exists():q['project_and_SDK_library_map_observation']=[line for line in Path('/proc/self/maps').read_text().splitlines()if any(v in line for v in ['_infinilm.cpython','libGLM','libinfinilm_runtime','libinfinirt.so','libinfiniops.so','libinfiniccl.so'])]
  if 'torch'in sys.modules:q['Torch_CUDA_initialized']=sys.modules['torch'].cuda.is_initialized()
  (B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in['passed','actual_CPU','elapsed_s','error','actual_unique_encodings','planned_multigroup_by_suite']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
