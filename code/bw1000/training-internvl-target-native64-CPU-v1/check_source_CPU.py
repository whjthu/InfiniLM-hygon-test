"""New blocked-admission/deadline source boundaries only; never import target worker."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

D=Path(__file__).resolve().parent


def main():
    for path in D.rglob('*.py'):ast.parse(path.read_bytes())
    spec=importlib.util.spec_from_file_location('target_scope_only',D/'scope.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    claim=module.read(D/'CLAIM.json');pre=module.read(D/'PREREQUISITES.json')
    rejected=[]
    def bad(name,claim,pre):
        try:module.admission(claim,pre)
        except ValueError:rejected.append(name)
        else:raise AssertionError('blocked prerequisite admitted '+name)
    bad('delivered_root_GO_false_before_SSH_or_Torch',claim,pre)
    c=dict(claim,CPU_GO=True,root_GO=True);p=dict(pre,root_GO=True)
    bad('pending_Torch_permission_no_import_probe',c,p)
    p=dict(p,pending_Torch_version_permission=False)
    bad('sixshadow_original_UNKNOWN_no_new_loader',c,p)
    p=dict(p,sixshadow_original_state='LEGITIMATELY_RESOLVED_NO_REPLAY')
    bad('Hygon_wrapper_policy_unknown_not_A100_qualification',c,p)
    valid=[]
    for remaining in (900,780,210):
        value=module.limits(remaining)
        assert value['Slurm_seconds']%60==0 and value['Slurm_seconds']+90<=remaining and value['work_seconds']+60<=value['Slurm_seconds']
        valid.append(dict(remaining=remaining,**value))
    for remaining in (161,100,0):
        try:module.limits(remaining)
        except ValueError:rejected.append('insufficient_original_budget_'+str(remaining))
        else:raise AssertionError('late Slurm step admitted')
    original=D.parents[2]/'code/bw1000/training-internvl-native-caller-CPU-v1/run_checks.py'
    def row_loop(path):
        return next(node for node in ast.walk(ast.parse(path.read_bytes()))
                    if isinstance(node,ast.For) and isinstance(node.target,ast.Tuple)
                    and [getattr(item,'id',None) for item in node.target.elts]==['ordinal','raw_row'])
    assert ast.dump(row_loop(original),include_attributes=False)==ast.dump(row_loop(D/'worker.py'),include_attributes=False)
    helper=D/'wrapper_binding.py';assert hashlib.sha256(helper.read_bytes()).hexdigest()=='d8c2111628c937d2141d35c095177adbc73f7d5bb8900c94dcc99e6a6d20d398'
    canonical=D/'canonical/first_update_audit.py';assert len(canonical.read_bytes())==6939 and hashlib.sha256(canonical.read_bytes()).hexdigest()=='816ef694537273ac04d8f7d40df5694a16e8a869797d2d2f65664d21e6a0ea49'
    assert 'torch' not in sys.modules and 'swift' not in sys.modules
    print(json.dumps(dict(passed=True,blocked_admission_rejections=rejected,valid_budget_boundaries=valid,
                         original_complete64_row_loop_AST_unchanged=True,helper_and_canonical_exact=True,
                         Torch_imports=0,SSH=0,GPU=0,payload_reads=0,actual_target_invocations=0,CPU_GO=False)))


if __name__=='__main__':
    main()
