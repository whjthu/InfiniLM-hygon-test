"""Check actual frozen source/data and synthetic runtime observations separately."""

import ast
import importlib.util
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

import evidence as e
import native_entry as native
import prepare as source


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BW=HERE.parent
PRIVATE=REPO/"local-only/specialized01-textvqa-original-train16-private-v1"


def denied(call, reason):
    try:
        call()
    except ValueError as failure:
        assert str(failure)==reason,(str(failure),reason)
        return reason
    raise AssertionError(reason)


def check():
    for file in HERE.glob("*.py"):
        ast.parse(file.read_bytes(),filename=file.name)
    deps={name:importlib.util.find_spec(name) is not None for name in ("torch","accelerate","transformers","swift")}
    frozen=BW/"training-qwen-A100-native-contract-CPU-v1/framework"
    sys.path.insert(0,str(frozen))
    try:
        builder=native.load(BW/"training-qwen-target-first-update-recipe-v1/recipe.py",
                            source.RECIPE_SHA,"pinned_original_recipe").prepare
    finally:
        sys.path.pop(0)
    native_pass=REPO/"results/bw1000/2026-11-25/specialized01-textvqa-qwen-native-1251-41-training-inputs-pass.json"
    original220=(REPO/"results/bw1000/2026-11-14"/
                 "training-qwen-1087656-220-native25-independent-terminal-and-checkpoint-inventory.json")
    assets=BW/"training-qwen-native25-hip-v3/ASSETS.json"
    binding=BW/"training-qwen-target-first-update-binding-v1/QWEN_TARGET_NATIVE_BINDING.json"
    original=e.read(BW/"training-qwen-A100-first-updates-v1/original-launch.json",32768,source.ORIGINAL_SHA)
    production=source.history(native_pass,original220,assets,binding)
    run=e.BASE/("qwen-textvqa-first-two-"+"e"*32)
    framework=e.BASE/"training-qwen-a68952c-swift-initial-CPU-v1/framework"
    plugin=run/"code/street_updates.py"
    launch=source.prepare(builder,original,binding,framework,run,plugin,29900,epoch=100,deadline=200,maximum=100)
    argv=launch["command"]
    assert argv[source.value(argv,"--dataset")]==str(e.DATASET) and launch["cwd"]==str(e.DATASET.parent)
    assert [argv[source.value(argv,k)] for k in ("--max_steps","--gradient_accumulation_steps","--save_strategy")]==["25","4","no"]
    assert "--val_dataset" not in argv and "--resume_from_checkpoint" not in argv and launch["Model_GO"] is False
    assert launch["environment_overrides"]["NPROC_PER_NODE"]=="8"
    assert production["model_member_count"]==14 and launch["initial_trainables_SHA256"]==source.INIT_NAMED_SHA

    real_rows=e.dataset_rows(PRIVATE/"TRAIN_SWIFT.jsonl",PRIVATE/"MEDIA_PINS.json")
    data=(PRIVATE/"TRAIN_SWIFT.jsonl").read_bytes().splitlines()
    roles={"VQA":0,"OCR":0}
    report=[]
    for i,line in enumerate(data):
        task="OCR" if json.loads(line)["messages"][0]["content"].startswith("<image>\nTranscribe the readable words ") else "VQA"
        roles[task]+=1
        # No Swift or Torch in this local runtime. Synthetic token fingerprints
        # below are solely for testing the runtime lookup and receipt boundaries.
        report.append(dict(row=i,task=task,input_ids_SHA256=e.pin([i,2]),labels_SHA256=e.pin([-100,i]),
                           tokens=2,supervised_tokens=1,grid=[[1,2,2]]))
    assert roles==e.TASKS
    index=e.fingerprints(report,real_rows)
    order=random.Random(42).sample(range(41),41)
    padded=order+order[:7]
    ranks={}
    for rank in range(8):
        records={}
        for step,(begin,end) in enumerate(((0,4),(4,6)),start=1):
            rows=[]
            for seen in range(begin,end):
                slot=e.schedule_position(rank,seen)
                i=padded[slot["global_slot_in_epoch"]]
                row=e.identify(index,[i,2],[-100,i],[1,2,2])
                assert row["record_SHA256"]==real_rows[i]
                rows.append({**row,**slot})
            records[step]={"rank":rank,"global_step":step,"parameters_after_SHA256":str(step)*64,
                           "batches":rows,"parameter_bytes_changed":step==2,"counter_advance":True}
        ranks[rank]=records
    summary=e.summarize(ranks)
    assert summary["qualified_nonzero_update"]==2 and summary["observed_slots"]==48
    assert summary["distinct_original_rows"]==41 and summary["repeated_slots"]==7
    assert [row["actual_examples"] for row in summary["updates"]]==[32,16]
    assert e.schedule_position(0,6)["epoch"]==1
    assert e.summarize({0:{1:ranks[0][1]}})["qualified_nonzero_update"] is None
    denials=[denied(lambda:e.identify(index,[0,2],[-100,0],[1,1,1]),"ACTUAL_BATCH_HAS_NO_EXACT_ORIGINAL_ROW"),
             denied(lambda:e.fingerprints([report[0]]+[{**report[1],
                  "input_ids_SHA256":report[0]["input_ids_SHA256"],"labels_SHA256":report[0]["labels_SHA256"]}]
                  +report[2:],real_rows),"NATIVE_ROW_FINGERPRINT_MISSING_OR_AMBIGUOUS")]
    invalid={**ranks[7][2],"batches":[*ranks[7][2]["batches"][:-1],dict(ranks[7][2]["batches"][-1],row=40)]}
    if invalid["batches"][-1]["row"]==ranks[7][2]["batches"][-1]["row"]:
        invalid["batches"][-1]["row"]=39
    denials.append(denied(lambda:e.summarize({**ranks,7:{1:ranks[7][1],2:invalid}}),
                          "NATIVE_SAMPLER_PADDING_OR_PERMUTATION_DIVERGED"))
    assert native.capture(REPO/"local-only"/"nonexistent-r19-fixture")["planned_optimizer_updates"]==2
    node="f0r0n0"
    cards=[f"0000:{i:02x}:00.0" for i in range(8)]
    report=dict(step="1.7",node=node,PCI_by_index=cards,leased_snapshot=dict(node=node,pci=cards,boot_id="fixture",
                kfd_holder_count=0,full_fd_inventory_performed=False,monotonic_ns=100))
    env=dict(SLURM_JOB_ID="1",SLURM_STEP_ID="7",SLURM_CPUS_PER_TASK="64")
    assert native.assigned(report,"1",node,env,node,"fixture",range(64),120)=="1.7"
    denials.append(denied(lambda:native.assigned(report,"1",node,env,node,"fixture",range(64),46_000_000_000),
                          "EXACT_EXISTING_ASSIGNED_EIGHT_DCU_STEP_REQUIRED"))
    captured=[]
    with TemporaryDirectory(dir=REPO/"local-only") as tmp:
        root=Path(tmp)
        owned=native.load(BW/"training-qwen-native25-hip-v3/owned_processes.py",native.OWNED_SHA,"exact_owned")
        owned.snapshot=lambda:{}
        def child(argv,**kwargs):
            assert argv[2:]==launch["command"] and kwargs["cwd"]==launch["cwd"] and kwargs["start_new_session"]
            captured.append(True)
            return SimpleNamespace(pid=1,poll=lambda:0,returncode=0)
        result=native.run_owned(root,launch,owned,time.time()+4,start=child)
        assert result["child_returncode"]==0 and result["cleanup"]=={"owned_processes_remaining":0,"termination_needed":False}
    return {"state":"ORIGINAL_RECIPE_AND_STREET_SCENE_ROWS_WITH_RANK_FIXTURES_CHECKED",
            "context":4096,"model_files_pinned":production["model_member_count"],
            "model_manifest_SHA256":production["model_manifest_SHA256"],
            "initial_trainables_SHA256":source.INIT_NAMED_SHA,"dataset_SHA256":e.DATA_SHA,
            "native_input_terminal_SHA256":e.NATIVE_PUBLIC_SHA,"local_dependencies_present":deps,
            "actual_local_frozen_rows":len(real_rows),"frozen_row_cases":roles,"local_native_sampler_rows":0,
            "local_optimizer_updates":0,"fixture_sampler":summary,"fixture_launches_captured":len(captured),
            "rejected":denials,"remote_calls":0,"Git":0,"Slurm":0,"GPU":0}


if __name__=="__main__":
    print(json.dumps(check(),sort_keys=True))
