"""One Linux CPU closure: real entry with recording providers, actual static pins."""
from pathlib import Path
import ast,hashlib,inspect,json,os,sys,time,traceback,tempfile
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Kimi-long-target-bootstrap-r346');B.mkdir(exist_ok=True);assert not(B/'CPU-result.json').exists()
start=time.monotonic();q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,actual_Native_Context=False,checkpoint_payload_read=False,source_manifest_SHA256=hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
try:
 assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 assert q['source_manifest_SHA256']==os.environ['KIMI_R346_CPU_MANIFEST_SHA']
 from audit import verify_bundle,verify_artifacts
 verify_bundle(D,q['source_manifest_SHA256'])
 for p in D.glob('*.py'):ast.parse(p.read_text())
 assert all(b'\r\n'not in p.read_bytes()for p in D.glob('*.sh'))
 from check_entry_CPU import run,environment,FaultGate
 q['whole_bootstrap_recording_entry']=run()
 from lifecycle import Settings
 with tempfile.TemporaryDirectory()as temporary:
  root=Path(temporary);(root/'control').mkdir();settings=Settings(environment(root),root);bound=verify_artifacts(D,FaultGate(settings))
  q['actual_source_and_receipt_audit']=bound['audit'];m=bound['modules'];frozen=json.loads((bound['driver']/'frozen_inputs.json').read_bytes())
  m['pair_driver']._validate_fixtures(frozen)
  inspect.signature(m['pair_driver'].run_pair).bind(None,frozen,None,None,deadline=1,cancelled=lambda:False,save=lambda x:None,enabled=True)
  inspect.signature(m['capture_transport'].establish).bind(None,'127.0.0.1',45947,'CPU-static-source-identity-000000000',1)
  assert m['long_capture_native'].service_geometry(2,2,1)=={'target_collective_cards':32,'external_draft_cards':1,'all_service_cards':33}
  q['real337_callable_and_frozen_inputs_bound']=True
 assert 'torch'not in sys.modules and'infinilm'not in sys.modules,'CPU closure imported a GPU runtime unexpectedly'
 q.update(completed=True,passed=True)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
finally:
 q['elapsed_s']=time.monotonic()-start;(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q),flush=True)
assert q['completed']and q['passed']
