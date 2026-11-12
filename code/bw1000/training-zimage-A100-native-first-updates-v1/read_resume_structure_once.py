"""CPU-only saved-state diagnosis;no model replay or gradient recomputation."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
code=b'''from pathlib import Path
import json,os,signal,sys
signal.alarm(90);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
w=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-A100-native-first-updates-v1')
import torch
torch.set_num_threads(1);assert not torch.cuda.is_initialized()
sys.path.insert(0,str(w/'framework/generation/src'))
from gencuda import zimage_training_resume_checkpoint as resume
from gencuda import hy_resume_checkpoint as common
c20=resume.load_checkpoint(w/'continuous/checkpoint-20/state.private.pt');c25=resume.load_checkpoint(w/'continuous/checkpoint-25/state.private.pt');r25=resume.load_checkpoint(w/'resume20/checkpoint-25/state.private.pt')
false_modes=[n for n,value in c20['state']['modes'].items() if not value]
components={n:common.tree_sha256(c25['state'][n])==common.tree_sha256(r25['state'][n]) for n in c25['state']}
groups=[{k:v for k,v in g.items() if k!='params'} for g in c20['state']['optimizer']['param_groups']]
changed={};step_values={};dtypes=set()
for key,values in c25['state']['optimizer']['state'].items():
 other=r25['state']['optimizer']['state'][key]
 for name,t in values.items():
  if isinstance(t,torch.Tensor):
   changed[name]=changed.get(name,0)+int(not torch.equal(t,other[name]));dtypes.add((name,str(t.dtype),str(t.device)))
   if name=='step':step_values[str(float(t))]=step_values.get(str(float(t)),0)+1
assert not torch.cuda.is_initialized()
print(json.dumps(dict(passed=True,saved20_false_mode_count=len(false_modes),saved20_false_modes=false_modes[:40],
 forcing_train_after_restore_can_change_saved_modes=bool(false_modes),final_state_component_equal=components,
 optimizer_groups_without_parameter_ids=groups,optimizer_changed_tensors_by_slot=changed,optimizer_step_counts=step_values,optimizer_state_dtypes=sorted(dtypes),
 replayed_GPU_or_model_runs=0,gradient_recomputations=0,GPU_initialized=False,root_cause_qualified=False)))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','/home/whj/nvidia-baseline/env-zimage/bin/python -B -u -'],input=code,capture_output=True,timeout=110)
(D/'resume-structure.stdout.private.json').write_bytes(r.stdout);(D/'resume-structure.stderr.private.txt').write_bytes(r.stderr)
v=json.loads(r.stdout) if r.stdout else None
with (D/'resume-structure-result.json').open('x') as f:json.dump(dict(transport_rc=r.returncode,value=v,source_SHA256=hashlib.sha256(code).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest()),f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,value=v)))
