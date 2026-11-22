"""Open the original Swift recipe and capture the first native microbatch metadata."""
import ctypes
import importlib.util
from importlib import metadata
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

import owned_processes
from scope import (ASSETS, BASELINE_SHA, CPU360, DATASET, FINGERPRINT_SHA, FRAMEWORK, HERE, INITIAL,
                   INITIAL_BYTES, INPUT_SHA, LAUNCH_SHA, NAMED_SHA, NATIVE, PORT, REF_INPUTS, REQUEST, RUN,
                   digest, input_roster, launch_recipe, need, pin, seconds_left, verify_shape, write_once)


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


def private_tmp(source):
    mount = Path("/tmp")
    need(source.is_absolute() and source.resolve(strict=True) == source and source.is_dir()
         and os.path.samestat(source.stat(), mount.stat()), "Private scratch bind differs")
    stats = os.statvfs(mount)
    available = stats.f_bavail * stats.f_frsize
    need(available >= 2**30 and stats.f_favail > 0, "Private scratch capacity differs")
    return dict(scratch_mount_verified=True, scratch_available_bytes=available,
                scratch_available_inodes=stats.f_favail)


def private_unix_socket():
    need(os.environ.get("TMPDIR") == tempfile.gettempdir() == "/tmp",
         "Swift multiprocessing must use the private short /tmp path")
    with Listener(family="AF_UNIX") as listener:
        address = Path(listener.address)
        count = len(os.fsencode(address))
        need(address.is_relative_to("/tmp") and count <= 107,
             "Private multiprocessing socket exceeds sockaddr_un capacity")
    return dict(short_unix_socket_verified=True, unix_address_bytes=count)


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


def run(epoch, plan_sha):
    deadline=epoch+NATIVE
    seconds_left(epoch,NATIVE)
    intent=json.loads(pin(RUN/"intent.json"))
    need(intent["request"]==REQUEST and intent["epoch"]==epoch and intent["plan_SHA256"]==plan_sha,
         "Native request, time origin or mount plan differs")
    scratch = {**private_tmp(RUN / "tmp"), **private_unix_socket()}
    model_view=validate_views(json.loads(pin(RUN/"node-model-plan.json",expected=plan_sha,cap=65536)))
    pin(DATASET,expected="1c9aaddad5b681b50a6ddfa79c08e30e4820ee6ffcdc81bd473d0c2e84b35a0f",extent=26817)
    pin(ASSETS/"plugins/swift_callback_nvidia_baseline_v2.py",expected=BASELINE_SHA,extent=4692)
    need(INITIAL.resolve(strict=True)==INITIAL and INITIAL.is_file() and INITIAL.stat().st_size==INITIAL_BYTES,
         "Original .362 rank00 initial file extent differs")
    verified=binding()
    new_binding=RUN/"binding.json"
    binding_sha=write_once(new_binding,verified)
    runtime=runtime_identity()

    write_once(RUN/"device-bootstrap.json",dict(request=REQUEST,step=intent["step"],
       phase="runtime_return_before_HIP",scratch=scratch,runtime=runtime))
    pci=physical_cards(intent["leased"]["pci"])
    with socket.socket() as check:
        check.bind(("127.0.0.1",PORT))
    from tools.hygon_vlm_recipe import build_launch
    launch=launch_recipe(build_launch,pin(HERE/"original-launch.json",expected=LAUNCH_SHA))
    env=os.environ.copy()
    need(not set(env).intersection(("RANK","LOCAL_RANK","WORLD_SIZE","LOCAL_WORLD_SIZE","GROUP_RANK","NNODES","NODE_RANK")),
         "Expected an original single Swift parent")
    env.update(launch["environment_overrides"])
    field_root = RUN / REQUEST
    field_root.mkdir(mode=0o700)
    env.update(MASTER_ADDR="127.0.0.1",PYTHONPATH=":".join((str(FRAMEWORK), "/swift-runtime/site", str(HERE / "frozen-v5"), str(HERE))),
               HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HOME=str(RUN/"home"),TMPDIR="/tmp",
               HF_HOME=str(RUN/"cache"),T03_FIRST2_EPOCH=str(epoch),T03_FIRST2_DEADLINE=str(deadline),
               T03_FIRST2_BINDING=str(new_binding),T03_FIRST2_BINDING_SHA256=binding_sha,
               T03_FIRST2_CAPTURE=str(field_root / "native-gates"),
               T03_FIELD_FROZEN_V5=str(HERE / "frozen-v5"),
               T03_FIELD_OUTPUT=str(field_root), T03_FIELD_REQUEST=REQUEST)
    argv=[sys.executable,"-B",*launch["command"]]
    write_once(RUN/"native-intent.json",dict(request=REQUEST,argv=argv,runtime_versions=runtime,model_view=model_view,
               PCI_by_visible_index=pci,actual_step=intent["step"],planned_update_slots=0,original_schedule_steps=25,
               planned_input_slots=8,binding_SHA256=binding_sha))
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
    from metadata_result import inspect_stop

    rc = child.returncode if child is not None else None
    report = inspect_stop(field_root, RUN / "native.log", verified["input_fingerprints_by_rank"],
                          returncode=rc, controller_error=error, cleanup=cleanup)
    report.update(actual_step=intent["step"], cleanup=cleanup)
    write_once(RUN/"native-outcome.json",report)
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    return int(not report["diagnostic_complete"])


if __name__=="__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--runtime-preflight":
        scratch = {**private_tmp(Path(sys.argv[2])), **private_unix_socket()}
        result = {**runtime_identity(), **scratch}
        need(result["torch_device_initialized"] is False, "Unexpected device initialization in runtime preflight")
        print(json.dumps(result, sort_keys=True))
    else:
        raise SystemExit(run(float(sys.argv[1]),sys.argv[2]))
