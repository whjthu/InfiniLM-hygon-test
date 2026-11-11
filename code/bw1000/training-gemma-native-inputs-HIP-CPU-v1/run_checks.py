"""Actual HIP-container CPU input comparison; no model or device execution."""
from pathlib import Path
import datetime,hashlib,importlib.util,io,json,os,signal,stat,subprocess,sys,time,unittest
W=Path('/evidence');source=Path('/input-framework');stage_source=Path('/input-stage');sys.path.insert(0,str(source));start=time.monotonic();epoch=float(os.environ['INPUT_CPU_EPOCH']);deadline=float(os.environ['INPUT_CPU_DEADLINE']);initial=deadline-time.time();assert 0<initial<=220
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original input CPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,initial)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def left(reserve=0):
 value=min(deadline-time.time(),initial-(time.monotonic()-start))-reserve
 if value<=0:raise TimeoutError('original input CPU budget exhausted')
 return value
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']);assert owner['node']=='a14r1n17' and os.environ['SLURM_CPUS_PER_TASK']=='4';dump(W/'owner.json',owner)
result=None;stage='source'
try:
 mf=json.loads((stage_source/'SOURCE_FILES.json').read_bytes())
 for name,pin in mf['files'].items():assert (stage_source/name).stat().st_size==pin['bytes'] and sha(stage_source/name)==pin['SHA256']
 fm=json.loads((stage_source/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==311 and fm['framework_commit']=='f082cb6bd841cf02f3d4afb0700c15c00f033a0e'
 for name,pin in fm['files'].items():
  left();assert (source/name).stat().st_size==pin['bytes'] and sha(source/name)==pin['SHA256'] and stat.S_IMODE((source/name).stat().st_mode)==pin['mode']
 deps=json.loads((stage_source/'DEPENDENCIES.json').read_bytes());model=Path('/gemma-assets/models/gemma-3-4b-it');pins=json.loads((stage_source/'MODEL_PROCESSOR_FILES.json').read_bytes());assert len(pins['files'])==14 and sum(x['bytes'] for x in pins['files'].values())==39352681
 for name,pin in pins['files'].items():assert (model/name).stat().st_size==pin['bytes'] and sha(model/name)==pin['SHA256']
 assert set(p.name for p in model.iterdir())==set(pins['files'])
 assets=json.loads((stage_source/'ORIGINAL_INPUT_FILES.json').read_bytes());root=Path(deps['inputs']['host_root']);assert len(assets['files'])==67
 for pin in assets['files']:assert (root/pin['name']).stat().st_size==pin['bytes'] and sha(root/pin['name'])==pin['SHA256']
 dataset=Path(deps['inputs']['host_dataset']['path']);assert dataset.stat().st_size==deps['inputs']['host_dataset']['bytes'] and sha(dataset)==deps['inputs']['host_dataset']['SHA256']
 remap={p['source']:str(root/p['name']) for p in assets['files']}
 def convert(v):
  if isinstance(v,str):return remap.get(v,v)
  if isinstance(v,list):return [convert(x) for x in v]
  if isinstance(v,dict):return {k:convert(x) for k,x in v.items()}
  return v
 original_rows=[json.loads(x) for x in (root/'original.cuda.jsonl').read_bytes().splitlines()];actual_rows=[json.loads(x) for x in dataset.read_bytes().splitlines()];assert len(original_rows)==len(actual_rows)==64 and convert(original_rows)==actual_rows
 stage='five-unit-tests';log=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('vlm.tests.test_template_inputs');tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(W/'tests.txt').write_text(log.getvalue());assert tests.wasSuccessful() and tests.testsRun==5
 stage='native-template-64';command=[sys.executable,'-B',str(source/'tools/hygon_vlm_inputs.py'),'T02.sft','--a100-launch',str(stage_source/'original-launch.json'),'--model-root',str(model),'--model-manifest',str(stage_source/'MODEL_PROCESSOR_FILES.json'),'--dataset',str(dataset),'--dataset-sha256',deps['inputs']['host_dataset']['SHA256'],'--output',str(W/'encoded'),'--epoch',str(epoch),'--deadline',str(deadline-15)]
 with (W/'capture.private.log').open('xb') as stream:r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=left(10))
 capture_path=W/'encoded/result.json'
 if r.returncode or not capture_path.is_file():
  fail=W/'encoded/failure.json';failure=json.loads(fail.read_bytes()) if fail.is_file() else {};raise RuntimeError('native capture failed')
 capture=json.loads(capture_path.read_bytes());assert capture['passed'] and capture['rows']==64 and capture['CUDA_initialized'] is False and capture['model_loaded'] is False and sha(W/'encoded/input-identities.json')==capture['identities_SHA256']
 stage='independent-all-field-comparison';original=stage_source/'ORIGINAL_REFERENCE.json';assert sha(original)==deps['original_CPU_reference']['identities_SHA256'];from vlm.runtime.template_inputs import compare_input_identities
 comparison=compare_input_identities(json.loads(original.read_bytes()),json.loads((W/'encoded/input-identities.json').read_bytes()));dump(W/'comparison.json',comparison)
 assert capture['settings']==deps['original_CPU_reference']['settings']
 import torch
 assert torch.version.hip and not torch.version.cuda and not torch.cuda.is_initialized()
 providers={}
 for name in ['torch','transformers','swift','tokenizers','PIL','numpy']:
  spec=importlib.util.find_spec(name);p=Path(spec.origin);providers[name]=dict(path=str(p),bytes=p.stat().st_size,SHA256=sha(p))
 result=dict(passed=comparison['all_fields_exact'],CPU_capture_completed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner,epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_tree=fm['framework_tree'],framework_files_verified=311,source_files_verified=len(mf['files']),source_manifest_SHA256=sha(stage_source/'SOURCE_FILES.json'),framework_manifest_SHA256=sha(stage_source/'FRAMEWORK_FILES.json'),original_asset_files_verified=67,path_rewrite_only_proved=True,processor_asset_files_verified=14,tests=5,tests_SHA256=sha(W/'tests.txt'),capture=capture,capture_result_SHA256=sha(capture_path),actual_runtime_providers=providers,HIP=torch.version.hip,Python=sys.version,comparison=comparison,comparison_SHA256=sha(W/'comparison.json'),original_reference_SHA256=sha(original),target_identities_SHA256=sha(W/'encoded/input-identities.json'),raw_encoded_payload_saved=False,GPU_cards=0,model_loaded=False,first_update_numerics_qualified=False,recovery_qualified=False,formal_performance=False,Model_GO=False,wall_seconds=time.monotonic()-start,qualification='Same-input native CPU preprocessing only. No distributed rank batch,model forward,update,25-step,resume,performance or effects acceptance.')
except BaseException as error:
 fail=W/'encoded/failure.json';result=dict(passed=False,owner=owner,epoch=epoch,deadline=deadline,stage=stage,error_type=type(error).__name__,capture_failure=json.loads(fail.read_bytes()) if fail.is_file() else None,wall_seconds=time.monotonic()-start,GPU_cards=0,Model_GO=False)
finally:
 dump(W/'CPU-result.json',result);print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
