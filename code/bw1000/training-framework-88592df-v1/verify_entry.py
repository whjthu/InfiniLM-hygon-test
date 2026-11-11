"""Verify only the shared source entry in the target container; no model imports."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
assert all(os.environ[n]=='-1' for n in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
epoch=float(os.environ['FRAMEWORK_EPOCH']);deadline=float(os.environ['FRAMEWORK_DEADLINE'])
assert deadline-epoch==90 and epoch<=time.time()<deadline
entry=Path('/shared-training/tools/hygon_training_recipe.py')
r=subprocess.run([sys.executable,'-B',str(entry),'list'],capture_output=True,text=True,timeout=min(30,max(1,deadline-time.time()-8)))
assert r.returncode==0,r.stderr[-1000:]
ids=json.loads(r.stdout)
expected=['T01.sft','T02.sft','T03.sft','T04.sft','T05.sft','T06.sft','T06.ddpo','T07.sft','T07.flow-grpo','T08.flow-matching-lora','T08.motion-rft','T09.sft','T09.dp2','T09.sage-grpo','T10.shape-flow-lora','T11.material-head','T12.camera-head']
assert ids==expected and len(set(ids))==17
assert 'torch' not in sys.modules
value={'observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'framework_commit':'88592dfb1a7ac70b217cd54eaac90b7dea2a179b','python':sys.version,'case_ids':ids,'entry_returncode':r.returncode,'qualification':'Source entry lists all17 cases in target container only; no model, training, restore, numerical, performance or held-out qualification.','GPU_allocation':0}
with Path('/evidence/entry-verification.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
