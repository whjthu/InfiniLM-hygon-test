"""Validate terminal gate and identity-only renderer with synthetic outcomes, without GPU."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

import training_guard as guard
from render_controller import controller, scorer, LABEL


def reject(action):
    try:
        action()
    except (ValueError,KeyError,TypeError):
        return
    raise AssertionError("Invalid terminal gate unexpectedly admitted")


def run(original_source):
    sys.path.insert(0,str(original_source))
    published="218d6e6f"+"0"*32
    binding=dict(schema="vggt-continuous25-dev3-training-binding-v1",original_request=guard.REQUEST,
                 original_publication=published,gpu_step=guard.GPU,post_step="1087656.999999",
                 files_SHA256={key:"a"*64 for key in guard.FILES},
                 head_file_SHA256=guard.HEAD_FILE_SHA256,head_module_SHA256=guard.HEAD_MODULE_SHA256,
                 head_stat=list(guard.HEAD_STAT),allow_recovery_failure=False)
    binding["files_SHA256"]["arms/continuous/native-result.json"]=guard.CONTINUOUS_RESULT_SHA256
    def row(number,phase):
        return [number,"fixture-"+phase,"COMPLETED","0:0","1",guard.NODE,"gres/dcu=1" if phase=="gpu" else "cpu=2"]
    rows={number:row(number,phase) for phase,number in (("pre",guard.PRE),("gpu",guard.GPU),("post",binding["post_step"]))}
    stages={phase:dict(row=rows[number],terminal_visible=True,passed=True) for phase,number in
            (("pre",guard.PRE),("gpu",guard.GPU),("post",binding["post_step"]))}
    docs={
        "outcome.json":dict(request=guard.REQUEST,publication=published,passed=True,stages=stages),
        "before.json":dict(pci=["fixture"],boot_id="boot"),
        "after.json":dict(pci=["fixture"],boot_id="boot",kfd_holder_count=0,full_fd_inventory_performed=True),
        "arms/continuous/receipt.json":dict(result_SHA256=guard.CONTINUOUS_RESULT_SHA256,arm="continuous",
            executed_optimizer_steps=25,final_state_SHA256="b"*64,checkpoint_identity={"data":"c"*64},
            steps=[dict(global_optimizer_step=i,batch_cursor=(i-1)%2) for i in range(1,26)]),
        "arms/continuous/native-result.json":dict(metrics=dict(global_optimizer_step=25,executed_optimizer_steps=25,
            completed_global_step_before_resume=0,measurement_steps=20,sequence_count=2,cached_valid_batches=2,
            camera_head_sha_after=guard.HEAD_MODULE_SHA256,final_state_sha256="b"*64,checkpoint_identity={"data":"c"*64}),
            checks={k:True for k in ("camera_head_weights_changed","frozen_aggregator_unchanged",
                    "nonzero_finite_camera_head_gradients","main_forward_device")},
            artifacts=[dict(path="/pilot/arms/continuous/train/camera_head.pt",sha256=guard.HEAD_FILE_SHA256)]),
    }
    assert guard.verify(binding,docs,rows)["original_recovery_passed"]
    cases=0
    for fault in ("running","postfail","postabsent","headartifact","headmodule","partial25","badcursor","foreignrequest","unclean"):
        bad=deepcopy(docs)
        current=deepcopy(rows)
        if fault=="running":
            bad["outcome.json"]["stages"]["gpu"]["row"][2]="RUNNING"
            current[guard.GPU][2]="RUNNING"
        elif fault=="postfail":
            bad["outcome.json"]["stages"]["post"]["passed"]=False
        elif fault=="postabsent":
            current.pop(binding["post_step"])
        elif fault=="headartifact":
            bad["arms/continuous/native-result.json"]["artifacts"][0]["sha256"]="d"*64
        elif fault=="headmodule":
            bad["arms/continuous/native-result.json"]["metrics"]["camera_head_sha_after"]="d"*64
        elif fault=="partial25":
            bad["arms/continuous/native-result.json"]["metrics"]["executed_optimizer_steps"]=24
        elif fault=="badcursor":
            bad["arms/continuous/receipt.json"]["steps"][0]["batch_cursor"]=1
        elif fault=="foreignrequest":
            bad["outcome.json"]["request"]="0"*32
        else:
            bad["after.json"]["kfd_holder_count"]=1
        reject(lambda:guard.verify(binding,bad,current))
        cases+=1
    failed=deepcopy(docs)
    failed["outcome.json"]["passed"]=False
    reject(lambda:guard.verify(binding,failed,rows))
    allowed=dict(binding,allow_recovery_failure=True)
    accepted=guard.verify(allowed,failed,rows)
    assert accepted["original_recovery_passed"] is False and accepted["recovery_failure_explicitly_accepted"] is True
    cases+=1
    original=(original_source/"score_vggt_tum_pair.py").read_bytes()
    rendered=scorer(original)
    before,after=ast.parse(original),ast.parse(rendered)
    stable=("angular_errors","relative_pose","summarize_development")
    for name in stable:
        old=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
        new=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(old)==ast.dump(new)
    assert rendered.count(b"model = VGGT(")==1 and rendered.count(b"raw_state = torch.load(weights,")==1
    body,_=controller((original_source/"run_tum_native_once.py").read_bytes(),binding)
    ast.parse(body)
    assert LABEL.encode() in body and b'training-vggt-tum-continuous25-recovery-v1' in body
    assert b'guard["preflight"](training_binding)' in body and body.index(b'guard["preflight"]')<body.index(b"    root.mkdir()")
    assert b"first-update-v2'/'train-output" not in body
    for path in Path(__file__).resolve().parent.glob("*.py"):
        ast.parse(path.read_bytes())
    return dict(state="LOCAL_CPU_DEV3_GATE_AND_RENDER_PASS",fixture_only=True,negative_gates_rejected=cases,
                score_math_AST_unchanged=list(stable),one_model_constructor=True,
                explicit_recovery_failure_preserved=True,actual_training_gate_executed=False,
                final_binding_ready=False,DCU_invocations=0)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_source",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.original_source)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,sort_keys=True)
        stream.write("\n")
    print(json.dumps(result,sort_keys=True))
