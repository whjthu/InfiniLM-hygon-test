"""Verify mounted inputs, run the ordinary Swift recipe, and close its two-update diagnostic."""
import ctypes
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import owned_processes
from scope import (ASSETS, BASELINE_SHA, CPU360, DATASET, FINGERPRINT_SHA, FRAMEWORK, HERE, INITIAL,
                   INITIAL_BYTES, INPUT_SHA, LAUNCH_SHA, NAMED_SHA, NATIVE, PORT, REF_INPUTS, REQUEST, RUN,
                   digest, finish_step, input_roster, launch_recipe, need, pin, seconds_left, verify_shape, write_once)


def qualified_framework():
    raw=pin(FRAMEWORK.parent/"source/FRAMEWORK_FILES.json",extent=112315,
            expected="e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41")
    manifest=json.loads(raw)
    need(len(manifest["files"])==393 and manifest["framework_commit"]=="a68952cb2285d6593b665147249d14ec1f8cecbc",
         "Deployed unified framework differs")
    members=("tools/hygon_vlm_recipe.py","vlm/runtime/swift_entry.py","vlm/runtime/first_update_capture.py",
             "vlm/runtime/first_update_audit.py","vlm/runtime/native_recipe_contract.py",
             "vlm/runtime/initial_trainables.py","vlm/runtime/checkpointing_lifecycle.py",
             "vlm/runtime/native_caller_budget.py","vlm/runtime/provider_sources.py","vlm/plugins/native_first_updates.py")
    for name in members:
        expected=manifest["files"][name]
        pin(FRAMEWORK/name,expected=expected["SHA256"],extent=expected["bytes"])
    need(manifest["files"]["vlm/runtime/first_update_audit.py"]["SHA256"]==FINGERPRINT_SHA,
         "Native input fingerprint provider differs")


def validate_views(plan):
    path=Path("/audit/assemble.py")
    raw=pin(path,expected="ebff2c2df8665cde980758aa0a8bfb06e5edf65cd71d1346bde2fc432315d3d6")
    module=importlib.util.module_from_spec(importlib.util.spec_from_file_location("internvl_inventory",path))
    exec(compile(raw,str(path),"exec"),module.__dict__)
    return module.verify_mounted(plan,module.readonly_mounts(Path("/proc/self/mountinfo").read_text()))


def binding():
    qualified_framework()
    sys.path.insert(0, str(FRAMEWORK))
    from vlm.runtime.native_recipe_contract import target_signature

    base=json.loads(pin(HERE/"NATIVE_BINDING.json",
                       expected="44fbf41909f19368f809a2d69b94e5874e46602b965936808a1e768a00d9ec1b"))
    target=json.loads(pin(HERE/"TARGET_MATH_SOURCES.json",
                       expected="5ea1da3481ab5bec153870613d1758a41db52f6f566d3b0dd2923900c19b0ed2"))
    actual=json.loads(pin(CPU360/"native-PEFT-structure.json",extent=83479,
                         expected="1b3c5d5ca3a36b8084c2fa51964f6eeea7f841eafc78805d98c8c82bec7dc308"))
    verify_shape(base,actual,target_signature)
    base["model_source_SHA256"]={key:value["SHA256"] for key,value in target["target_sources"].items()}
    need(base["model_source_SHA256"]==actual["measured_target_model_source_SHA256"],
         "Current target math source does not match actual .360")
    base["native_checkpointing_lifecycle"]["provider"]=actual["lifecycle"]["provider"]
    base["resolved_template_type"]="internvl_hf"
    base["capture_budget_conditions"]=dict(case="T03.sft",hidden_size=4096,vocab_size=151936,
        language_cache=False,hidden_state_stacks=False,attention_maps=False)
    base["initial_trainables_SHA256"]=NAMED_SHA
    base["input_fingerprints_by_rank"]=input_roster(json.loads(pin(REF_INPUTS,expected=INPUT_SHA,extent=25357)))
    return base


def verify_runtime_versions(observed, torch_version, hip_version, transformers_version):
    expected = {
        "torch": "2.10.0+das.opt1.dtk2604.2606161421.g91a580",
        "transformers": "5.5.0",
        "ms-swift": "4.4.2",
        "peft": "0.19.1",
        "accelerate": "1.14.0",
        "tokenizers": "0.22.2",
    }
    need(observed == expected and (torch_version, hip_version, transformers_version) ==
         ("2.10.0", "6.3.26113", "5.5.0"), "Qualified target Python runtime differs")


def runtime_identity():
    versions = {name: metadata.version(name) for name in
                ("torch", "transformers", "ms-swift", "peft", "accelerate", "tokenizers")}
    import torch
    import transformers

    verify_runtime_versions(versions, str(torch.__version__), torch.version.hip, str(transformers.__version__))
    distribution = Path(metadata.distribution("transformers").locate_file("transformers/__init__.py"))
    package = Path(transformers.__file__)
    need(package == distribution and package.is_absolute(), "Transformers module and distribution differ")
    pin(package, extent=40505, expected="460fadeb41958b1b47d7df8769b5f0ca3f46554ab733f52cf03a2b92d55d899a")
    files = TARGET_MATH_SOURCES
    for relative, expected in files.items():
        pin(package.parent/relative, extent=expected["bytes"], expected=expected["SHA256"])
    return dict(versions=versions, target_math_sources=files, torch_attr=str(torch.__version__),
                hip_attr=torch.version.hip, torch_device_initialized=torch.cuda.is_initialized())


TARGET_MATH_SOURCES = {
  "models/internvl/modeling_internvl.py": {
      "SHA256": "632fc3b219cd1d08c322e786fa25d7dbf07059f6ccb6fe22dd1c45592415ad9a",
      "bytes": 37559,
  },
  "models/qwen3/modeling_qwen3.py": {
      "SHA256": "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab",
      "bytes": 23353,
  },
}


def physical_cards(expected_pci):
    hip=ctypes.CDLL("libamdhip64.so")
    hip.hipGetDeviceCount.argtypes=[ctypes.POINTER(ctypes.c_int)]
    hip.hipDeviceGetPCIBusId.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
    count=ctypes.c_int()
    need(hip.hipGetDeviceCount(ctypes.byref(count))==0 and count.value==8,"HIP sees a different device count")
    cards=[]
    for idx in range(8):
        value=ctypes.create_string_buffer(32)
        need(hip.hipDeviceGetPCIBusId(value,len(value),idx)==0,"HIP physical PCI lookup failed")
        pci=value.value.decode("ascii").lower()
        if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",pci):
            pci="0000:"+pci
        cards.append(pci)
    need(len(set(cards))==8 and sorted(cards)==sorted(expected_pci),"Logical HIP indices do not match assigned cards")
    return cards


def report_rank(root,rank,expected_inputs):
    terminal=json.loads(pin(root/"terminal.json",cap=65536))
    need((terminal["rank"],terminal["world_size"],terminal["request"],terminal["native_schedule_steps"])
         ==(rank,8,REQUEST,25) and terminal["rank00_trainable_initial_digest"]==NAMED_SHA
         and terminal["raw_tensor_bytes_written"]==0,"Native rank terminal identity differs")
    files=terminal["files"]
    finish_step(terminal["actual_stop_step"],updates=terminal["optimizer_updates"],inputs=terminal["micro_forwards"],
                outputs=sum(row["name"].endswith("-output.pt") for row in files),
                clips=terminal["clip_calls"],names=[row["name"] for row in files])
    reports={}
    for row in files:
        name=row["name"]
        raw=pin(root/(name.removesuffix(".pt")+".json"),expected=row["SHA256"],extent=row["bytes"],cap=65536)
        item=json.loads(raw)
        need(item["rank"]==rank and item["point"]==name and item["request"]==REQUEST,
             "Wrong rank capture metadata")
        if name.endswith("-input.pt"):
            ordinal=int(name.split("-")[1])
            need(item["native_input_SHA256"]==expected_inputs[ordinal-1],"Original per-rank input fingerprint differs")
        reports[name]=item["fields"]
    ordered=(reports["update-%02d-%s-optimizer.pt"%(i,side)]["parameters"]["SHA256"]
             for i in (1,2) for side in ("before","after"))
    before1,after1,before2,after2=ordered
    need(before1!=after1 and before2!=after2 and after1==before2,"Actual two parameter updates could not be verified")
    return dict(rank=rank,first_update_parameter_SHA256=after1,second_update_parameter_SHA256=after2,
                reference_input_fingerprint_slots=len(expected_inputs),capture_slots=25,updates=2)


def run(epoch, plan_sha):
    deadline=epoch+NATIVE
    seconds_left(epoch,NATIVE)
    intent=json.loads(pin(RUN/"intent.json"))
    need(intent["request"]==REQUEST and intent["epoch"]==epoch and intent["plan_SHA256"]==plan_sha,
         "Native request, time origin or mount plan differs")
    model_view=validate_views(json.loads(pin(RUN/"node-model-plan.json",expected=plan_sha,cap=65536)))
    pin(DATASET,expected="1c9aaddad5b681b50a6ddfa79c08e30e4820ee6ffcdc81bd473d0c2e84b35a0f",extent=26817)
    pin(ASSETS/"plugins/swift_callback_nvidia_baseline_v2.py",expected=BASELINE_SHA,extent=4692)
    need(INITIAL.resolve(strict=True)==INITIAL and INITIAL.is_file() and INITIAL.stat().st_size==INITIAL_BYTES,
         "Original .362 rank00 initial file extent differs")
    verified=binding()
    new_binding=RUN/"binding.json"
    binding_sha=write_once(new_binding,verified)
    runtime=runtime_identity()
    pci=physical_cards(intent["leased"]["pci"])
    with socket.socket() as check:
        check.bind(("127.0.0.1",PORT))
    from tools.hygon_vlm_recipe import build_launch
    launch=launch_recipe(build_launch,pin(HERE/"original-launch.json",expected=LAUNCH_SHA))
    env=os.environ.copy()
    need(not set(env).intersection(("RANK","LOCAL_RANK","WORLD_SIZE","LOCAL_WORLD_SIZE","GROUP_RANK","NNODES","NODE_RANK")),
         "Expected an original single Swift parent")
    env.update(launch["environment_overrides"])
    env.update(MASTER_ADDR="127.0.0.1",PYTHONPATH=str(FRAMEWORK)+":/swift-runtime/site:"+str(HERE),
               HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HOME=str(RUN/"home"),TMPDIR=str(RUN/"tmp"),
               HF_HOME=str(RUN/"cache"),T03_FIRST2_EPOCH=str(epoch),T03_FIRST2_DEADLINE=str(deadline),
               T03_FIRST2_BINDING=str(new_binding),T03_FIRST2_BINDING_SHA256=binding_sha,
               T03_FIRST2_CAPTURE=str(RUN/"capture"))
    argv=[sys.executable,"-B",*launch["command"]]
    write_once(RUN/"native-intent.json",dict(request=REQUEST,argv=argv,runtime_versions=runtime,model_view=model_view,
               PCI_by_visible_index=pci,actual_step=intent["step"],planned_update_slots=2,original_schedule_steps=25,
               planned_input_slots=64,binding_SHA256=binding_sha))
    registry={}
    child=None
    error=None
    def interrupt(*_):
        raise TimeoutError("Owned native step was interrupted")
    for signum in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):
        signal.signal(signum,interrupt)
    try:
        with (RUN/"native.log").open("xb") as log:
            child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True)
            prior=owned_processes.snapshot().get(child.pid)
            if prior:
                registry[child.pid]={key:prior[key] for key in ("PID","PPID","PGID","startticks","state")}
            while child.poll() is None:
                owned_processes.refresh(registry,RUN)
                try:
                    child.wait(timeout=min(seconds_left(epoch,NATIVE),2))
                except subprocess.TimeoutExpired:
                    continue
    except BaseException as failure:
        error=type(failure).__name__
    finally:
        cleanup=owned_processes.cleanup(registry,RUN,grace=8)
        if child is not None:
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                error=error or "OWNED_NATIVE_CHILD_NOT_REAPED"
        write_once(RUN/"owned-final.json",dict(ownership=registry,cleanup=cleanup))
    reports=[]
    attempted=[]
    for rank in range(8):
        output=RUN/"capture"/f"rank-{rank:02d}"
        partial=dict(rank=rank,completed_update_records=None,completed_input_records=None)
        if (output/"progress.json").is_file():
            try:
                row=json.loads(pin(output/"progress.json",cap=65536))
                need(row["rank"]==rank and 0<=row["completed_updates"]<=2 and 0<=row["inputs_saved"]<=8,
                     "Unexpected native partial progress")
                partial.update(completed_update_records=row["completed_updates"],completed_input_records=row["inputs_saved"])
            except (ValueError,KeyError,OSError,TypeError):
                error=error or "INVALID_NATIVE_PARTIAL_PROGRESS"
        attempted.append(partial)
        if (output/"terminal.json").is_file():
            try:
                reports.append(report_rank(output,rank,verified["input_fingerprints_by_rank"][str(rank)]))
            except (ValueError,KeyError,OSError,TypeError):
                error=error or "INVALID_NATIVE_RANK_RECEIPT"
    complete=(child is not None and child.returncode==0 and error is None and len(reports)==8
              and cleanup["owned_processes_remaining"]==0 and not cleanup["termination_needed"]
              and len({row["second_update_parameter_SHA256"] for row in reports})==1)
    report=dict(request=REQUEST,actual_step=intent["step"],native_returncode=child.returncode if child else None,
                error=error,engineering_first_two_complete=complete,planned_ranks=8,qualified_rank_receipts=len(reports),
                qualified_updates_per_rank=2 if complete else None,planned_input_slots=64,
                all_reference_input_fingerprint_slots=64 if complete else None,rank_receipts=reports,
                original_partial_records=attempted,cleanup=cleanup,
                A100_remaining_rank_RNG_available=False,formal17_qualified=False,training_effect_GO=False,
                numerical_A100_alignment_qualified=False,formal_performance=False,raw_capture_tensor_artifacts_written=0)
    write_once(RUN/"native-outcome.json",report)
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    return int(not complete)


if __name__=="__main__":
    if sys.argv[1:] == ["--runtime-preflight"]:
        result = runtime_identity()
        need(result["torch_device_initialized"] is False, "Unexpected device initialization in runtime preflight")
        print(json.dumps(result, sort_keys=True))
    else:
        raise SystemExit(run(float(sys.argv[1]),sys.argv[2]))
