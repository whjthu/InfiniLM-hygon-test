"""One source CPU-only Gemma native template reference from frozen shared code."""
from pathlib import Path
import base64,datetime,hashlib,json,os,signal,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-native-inputs-v1');epoch=time.time();start=time.monotonic();deadline=epoch+300
assert not W.exists();W.mkdir(mode=0o700)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
write('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,GPU_cards=0))
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original CPU template epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.time()-5))
stage='source';result=None
try:
 for name,pin in FILES.items():
  p=W/name;assert not Path(name).is_absolute() and '..' not in Path(name).parts;raw=base64.b64decode(pin['b64']);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 source=W/'framework';fm=json.loads((W/'FRAMEWORK_FILES.json').read_bytes())
 for name in fm['execution_closure']:
  pin=fm['files'][name];assert (source/name).stat().st_size==pin['bytes'] and sha(source/name)==pin['SHA256']
 assets=json.loads((W/'ORIGINAL_INPUT_FILES.json').read_bytes())
 for pin in assets['files']:
  p=Path(pin['source']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false')
 python='/home/whj/nvidia-baseline/env-vlm/bin/python';stage='five_evidence_tests';unit=subprocess.run([python,'-B','-m','unittest','vlm.tests.test_template_inputs','-v'],cwd=source,env=env,capture_output=True,timeout=min(30,deadline-time.time()-30));(W/'tests.private.txt').write_bytes(unit.stdout+unit.stderr);assert unit.returncode==0 and b'Ran 5 tests' in unit.stderr
 stage='original_native_template_64';command=[python,'-B',str(source/'tools/hygon_vlm_inputs.py'),'T02.sft','--a100-launch',str(W/'original-launch.json'),'--model-root','/mnt/data/nvidia-baseline-20261010/inputs/models/gemma-3-4b-it','--model-manifest',str(W/'MODEL_PROCESSOR_FILES.json'),'--dataset','/mnt/data/nvidia-baseline-20261010/inputs/vlmdata/vlm_report_coco64.cuda.jsonl','--dataset-sha256','0e476394e3b0117d90cfad8f89ccfafe1800917692e86efed84eaf182c453811','--output',str(W/'encoded'),'--epoch',str(epoch),'--deadline',str(epoch+260)]
 with (W/'capture.private.log').open('xb') as log:r=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(.1,epoch+270-time.time()))
 capture_path=W/'encoded/result.json'
 if not capture_path.is_file() or r.returncode!=0:
  fail=W/'encoded/failure.json';failure=json.loads(fail.read_bytes()) if fail.exists() else {};raise RuntimeError('Native template failed: '+str(failure.get('error_type','NO_RESULT')))
 capture=json.loads(capture_path.read_bytes());identities_raw=(W/'encoded/input-identities.json').read_bytes();assert len(identities_raw)<=2*1024**2 and hashlib.sha256(identities_raw).hexdigest()==capture['identities_SHA256'] and capture['rows']==64 and capture['passed'] and capture['model_loaded'] is False and capture['CUDA_initialized'] is False
 identities=json.loads(identities_raw);result=dict(passed=True,claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,framework_commit=fm['framework_commit'],framework_tree=fm['framework_tree'],installed_execution_closure=fm['execution_closure'],tests=5,original_input_files_verified=len(assets['files']),capture=capture,input_identities=identities,original_result_SHA256=sha(capture_path),GPU_cards=0,model_loaded=False,raw_encoded_payload_saved=False,wall_seconds=time.monotonic()-start,qualification='Original native CPU row-order pre-model template identities only; no distributed first batch,forward,update,restoration or performance acceptance.')
except BaseException as error:
 fail=W/'encoded/failure.json';result=dict(passed=False,claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,stage=stage,error_type=type(error).__name__,capture_failure=json.loads(fail.read_bytes()) if fail.exists() else None,GPU_cards=0,Model_GO=False,wall_seconds=time.monotonic()-start)
finally:write('outcome.json',result);print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
