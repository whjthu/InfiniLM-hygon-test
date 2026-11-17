"""New pure stdlib29scope and actualschedule boundaries only; no Torch/GPU/file bodies."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
from common import check_state_contract,need

D=Path(__file__).resolve().parent


def run():
    c=json.loads((D/'PINNED_STATE.json').read_bytes());assert len(check_state_contract(c))==29
    tree=ast.parse((D/'run_CPU_once.py').read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='schedule')
    scope={'math':math,'need':need};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual bounded schedule>','exec'),scope)
    schedule=scope['schedule'];one=schedule(1000.,1000.);late=schedule(1000.,1139.)
    assert one['deadline']==1180 and one['work_deadline']==1140 and one['supervisor_stop']==1150 and one['held_cleanup_deadline']==1162 and one['registry_cleanup_deadline']==1165 and one['alarm_deadline']==1170 and one['client_deadline']==1175
    assert late==one
    rejects=[]
    def reject(name,fun):
        try:fun()
        except ValueError:rejects.append(name);return
        raise AssertionError(name)
    x=deepcopy(c);del x['checkpoint']['20']['files']['rng_state_7.pth'];reject('missing_rankRNG',lambda:check_state_contract(x))
    x=deepcopy(c);x['checkpoint']['20']['files']['optimizer.pt']['bytes']=0;reject('zero_statebytes',lambda:check_state_contract(x))
    x=deepcopy(c);x['state_total_bytes']-=1;reject('wrong_aggregate_bytes',lambda:check_state_contract(x))
    x=deepcopy(c);x['initial']['path']=x['initial']['path'].replace('initial.pt','other.pt');reject('wrong_initialpath',lambda:check_state_contract(x))
    reject('late_expired',lambda:schedule(1000.,1140.));reject('future_epoch',lambda:schedule(1001.,1000.));reject('nonfinite_clock',lambda:schedule(1000.,float('nan')))
    return dict(passed=True,normal=3,rejected=len(rejects),rejected_cases=rejects,actual_savedstate_body_reads=0,actual_Torch_imports=0,actual_GPU_or_training_calls=0,old_native_or_science_tests_rerun=0)

if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
