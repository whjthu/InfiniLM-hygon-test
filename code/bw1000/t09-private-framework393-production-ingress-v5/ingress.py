"""One no-model, no-SDK production-image ingress proof for the exact private T09 tree."""
import argparse
import ast
import importlib
import importlib.util
import sysconfig
from hashlib import sha256
import importlib.machinery
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

from framework_binding import MANIFEST_SHA256, verify_framework
from mount_evidence import check_objects
from contract import exact_host_mount_context
from contract import (DENOMINATOR,MANIFEST,REQUEST,LABEL,SHARED,STAGE_REQUEST,STAGE_SHA,AUDIT,AUDIT_SHA,
                      inspect_tree,validate_manifest,need)


BACKEND_SHA="2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae"
REQUIREMENTS={"gencuda.backends.hunyuanvideo_training_walltime":
              "generation/src/gencuda/backends/hunyuanvideo_training_walltime.py"}
ENTRY="tools/hygon_training_recipe.py"


def establish_step(authority):
    reported=os.environ.get("SLURM_STEP_ID","")
    parent=authority["parent"]
    step=parent+"."+reported
    need(authority["request"]==REQUEST and authority["label"]==LABEL
         and re.fullmatch(r"[0-9]+",parent) is not None and re.fullmatch(r"[0-9]+",reported) is not None
         and os.environ.get("SLURM_JOB_ID")==parent and socket.gethostname().split(".")[0]==authority["node"],
         "EXACT_CURRENT_ZERO_GRES_NUMERIC_STEP")
    cgroups=Path("/proc/self/cgroup").read_text("ascii").splitlines()
    need(any({"job_"+parent,"step_"+reported}<=set(line.split(":",2)[-1].split("/")) for line in cgroups),
         "EXACT_ZERO_GRES_STEP_CGROUP")
    need(os.environ.get("SLURM_GPUS_ON_NODE","") in ("","0")
         and all(os.environ.get(k,"") in ("","none","NoDevFiles","-1") for k in (
         "SLURM_STEP_GPUS","CUDA_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES")),
         "NO_EXPOSED_GPU_STEP_ENVIRONMENT")
    need(0<=time.time()-authority["epoch"]<120,"ORIGINAL_BOUNDED_ZERO_GRES_EPOCH")
    return step


def pathfinder(name, search):
    parent=None
    locations=search
    for segment in name.split("."):
        parent=segment if parent is None else parent+"."+segment
        spec=importlib.machinery.PathFinder.find_spec(parent, locations)
        need(spec is not None,"PRODUCTION_PROVIDER_NOT_RESOLVED")
        locations=spec.submodule_search_locations
    need(spec.origin is not None,"PRODUCTION_PROVIDER_HAS_NO_REAL_SOURCE")
    return Path(spec.origin)


def metadata_providers(root, pins, search):
    # This proves resolution/compilation of the actual Python providers. It
    # does not import gencuda nor any module that could transitively load HIP.
    providers={}
    for name,rel in (("script_entry",ENTRY),*REQUIREMENTS.items()):
        path=root/rel
        if name!="script_entry":
            need(pathfinder(name,search)==path,"RESOLVED_SOURCE_ESCAPES_CANONICAL_OVERLAY")
        raw=path.read_bytes()
        digest=sha256(raw).hexdigest()
        need(path.resolve(strict=True)==path and rel in pins and len(raw)==pins[rel]["bytes"]
             and digest==pins[rel]["SHA256"],"ACTUAL_ENTRY_MODULE_SOURCE_PIN")
        if name.endswith("hunyuanvideo_training_walltime"):
            need(digest==BACKEND_SHA,"EXACT_794_UNPROFILED_BACKEND")
        compile(raw,str(path),"exec",dont_inherit=True)
        providers[name]=dict(relative_path=rel,bytes=len(raw),SHA256=digest,compiled=True,executed=False)
    return providers


MEASUREMENT="generation/src/gencuda/measurement.py"
PACKAGE_INIT="generation/src/gencuda/__init__.py"
MEASUREMENT_SHA="8a9bb4ec134b4545ef135db1bdef7874331234ed0a8a32b95bcac63462b1bb94"


def stdlib_only(raw):
    allowed={"__future__","collections","dataclasses","math","typing","time"}
    calls={"dataclass","frozenset","tuple"}
    tree=ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            need(all(x.name.split(".")[0] in allowed for x in node.names),"SAFE_TIMING_STDLIB_IMPORTS")
        elif isinstance(node,ast.ImportFrom):
            need(node.level==0 and node.module is not None and node.module.split(".")[0] in allowed,
                 "SAFE_TIMING_STDLIB_IMPORTS")
    def top_level(node):
        functions=(ast.FunctionDef,ast.AsyncFunctionDef)
        need(isinstance(node,(ast.Import,ast.ImportFrom,ast.Assign,ast.AnnAssign,ast.ClassDef,ast.Expr,*functions)),
             "SAFE_TIMING_STATIC_BOOTSTRAP")
        if isinstance(node,ast.ClassDef):
            need(not node.bases and not node.keywords,"SAFE_TIMING_PLAIN_CLASS")
            for body in node.body:
                top_level(body)
        if isinstance(node,ast.Expr):
            need(isinstance(node.value,ast.Constant),"SAFE_TIMING_DOCSTRING_ONLY")
        stack=[child for child in ast.iter_child_nodes(node)
               if not (isinstance(node,(*functions,ast.ClassDef)) and child in node.body)]
        while stack:
            item=stack.pop()
            if isinstance(item,ast.Call):
                need(isinstance(item.func,ast.Name) and item.func.id in calls,"SAFE_TIMING_BOOTSTRAP_CALL")
            stack.extend(ast.iter_child_nodes(item))
    for node in tree.body:
        top_level(node)


def safe_measurement(root,pins,search):
    # Never import the training CLI, walltime backend, Torch or HIP.
    result={"module":"gencuda.measurement","executed":False,"selected_only_if_statically_safe":True}
    try:
        files=((PACKAGE_INIT,"gencuda"),(MEASUREMENT,"gencuda.measurement"))
        for relative,name in files:
            actual=root/relative
            need(relative in pins and pathfinder(name,search)==actual and actual.resolve(strict=True)==actual,
                 "SAFE_TIMING_CANONICAL_PROVIDER")
            raw=actual.read_bytes()
            need(sha256(raw).hexdigest()==pins[relative]["SHA256"] and len(raw)==pins[relative]["bytes"],
                 "SAFE_TIMING_EXACT_PIN")
            if relative==MEASUREMENT:
                need(sha256(raw).hexdigest()==MEASUREMENT_SHA,"SAFE_TIMING_EXACT_PRIOR_PROVIDER")
            stdlib_only(raw)
        stdlib=Path(sysconfig.get_path("stdlib")).resolve(strict=True)
        for module in ("__future__","collections","dataclasses","math","typing","time"):
            if module in sys.modules:
                actual=getattr(sys.modules[module],"__file__",None)
                need(actual is None or Path(actual).resolve(strict=True).is_relative_to(stdlib),"SAFE_UNSHADOWED_STDLIB")
            else:
                spec=importlib.util.find_spec(module)
                need(spec is not None and (spec.origin in ("built-in","frozen") or
                     Path(spec.origin).resolve(strict=True).is_relative_to(stdlib)),"SAFE_UNSHADOWED_STDLIB")
        mod=importlib.import_module("gencuda.measurement")
        need(Path(mod.__file__)==root/MEASUREMENT,"SAFE_TIMING_ACTUALLY_IMPORTED_CANONICAL_SOURCE")
        result.update(executed=True,source_SHA256=MEASUREMENT_SHA,imports="stdlib-only",provider_functions_called=0)
    except (OSError,ValueError,TypeError,KeyError,ImportError) as exc:
        result.update(reason=type(exc).__name__)
    return result


def no_gpu_imports():
    return not any(name=="torch" or name.startswith(("torch.","hip.","rocm.","gencuda.backends."))
                    or name in ("hip","rocm") for name in sys.modules)


def inspect(authority, stage, source_manifest, *, root=SHARED, manifest=MANIFEST, step_check=establish_step,
            mount_check=check_objects, verify=verify_framework, providers=metadata_providers,
            safe_provider=safe_measurement, host=None):
    report=dict(request=REQUEST,phase="STEP",passed=False,planned=DENOMINATOR,before_verified=0,
                after_verified=0,providers_planned=1+len(REQUIREMENTS),providers_verified=0,
                model_loaded=False,weight_payload_bytes_read=0,GPU_SDK_calls=0,optimizer_updates=0,formal_performance=False)
    try:
        report["step"]=step_check(authority)
        need(host and host["request"]==REQUEST and host["step"]==report["step"] and host["passed"] is True
             and host["actual_host_cgroup"] is True and host["epoch"]==authority["epoch"]
             and host["label"]==authority["label"] and host["node"]==authority["node"]
             and exact_host_mount_context(host),
             "EXACT_SAME_STEP_HOST_CONTROLLER_TRES")
        need(sys.dont_write_bytecode is True and os.environ.get("PYTHONDONTWRITEBYTECODE")=="1"
             and sys.pycache_prefix is None,"NO_PYTHON_BYTECODE_CACHE_READ_PATH_CHANGE_OR_WRITES")
        report["phase"]="MOUNT"
        report["mount_evidence"]=mount_check()
        need(report["mount_evidence"]["passed"] is True,"EFFECTIVE_FD_MOUNT_ADMISSION_FAILED")
        need(stage["request"]==STAGE_REQUEST and stage["passed"] is True and stage["verified_original"]==DENOMINATOR
             and stage["manifest_SHA256"]==MANIFEST_SHA256,"EXACT_NEW_PRIVATE_STAGE_RESULT")
        pins=validate_manifest(manifest.read_bytes())
        report["phase"]="ORIGINAL_BEFORE"
        original_before=verify(root,manifest)
        before=inspect_tree(root,pins,category="PRIVATE_CANONICAL_FRAMEWORK")
        report.update(before_verified=original_before["framework_files"],before_inventory=before)
        report["phase"]="PROVIDERS"
        report["providers"]=providers(root,pins,sys.path)
        report["providers_verified"]=len(report["providers"])
        report["safe_timing_provider"]=safe_provider(root,pins,sys.path)
        report["phase"]="ORIGINAL_AFTER"
        original_after=verify(root,manifest)
        after=inspect_tree(root,pins,category="PRIVATE_CANONICAL_FRAMEWORK")
        report.update(after_verified=original_after["framework_files"],after_inventory=after)
        need(before==after==stage["staged_inventory"] and original_after==original_before==stage["original_verifier"]
             and report["before_verified"]==report["after_verified"]==DENOMINATOR,"EXACT_UNCHANGED_FRAMEWORK_INVENTORY")
        need(no_gpu_imports(),"NO_ACTUAL_TORCH_OR_GPU_PROVIDER_IMPORTED")
        report.update(phase="COMPLETE",passed=True,bound_source_manifest_SHA256=source_manifest,readonly_overlay=True,
                      private_same_tree_and_manifest=True,actual_same_step_host_gate=True,production_CLI_imported=False,
                      canonical_stage_SHA256=STAGE_SHA,canonical_stage_audit_SHA256=AUDIT_SHA)
    except (OSError,ValueError,TypeError,KeyError,SyntaxError) as exc:
        report.update(error_class=type(exc).__name__,errno=getattr(exc,"errno",None),
                      failure_code=str(exc) if type(exc) is ValueError and re.fullmatch("[A-Z0-9_]{1,80}",str(exc)) else None)
    return report


def pinned(path, sha, cap=65536):
    need(re.fullmatch(r"[0-9a-f]{64}",sha) is not None and path.resolve(strict=True)==path and path.is_file(),
         "CANONICAL_BOUND_METADATA")
    with path.open("rb") as stream:
        raw=stream.read(cap+1)
    need(0<len(raw)<=cap and sha256(raw).hexdigest()==sha,"BOUND_METADATA_DIGEST")
    return json.loads(raw)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("stage-sha","authority-sha","source-manifest-sha"):
        p.add_argument("--"+name,required=True)
    args=p.parse_args()
    scope=Path("/t09-ingress-scope")
    authority=pinned(scope/"authority.json",args.authority_sha)
    stage=Path("/t09-stage-ro")
    need(args.stage_sha==STAGE_SHA,"SHARED_SEALED_STAGE_SHA_EXACT")
    staged=pinned(stage/"stage-result.json",args.stage_sha)
    with AUDIT.open("rb") as journal:
        audit=journal.read(65537)
    need(len(audit)<=65536 and sha256(audit).hexdigest()==AUDIT_SHA,"EXACT_SEALED_STAGE_AUDIT")
    host=(scope/"host-step.json").read_bytes()
    need(0<len(host)<=16384,"EXACT_BOUNDED_HOST_GATE")
    host_record=json.loads(host)
    need(host_record["authority_SHA256"]==args.authority_sha,"EXACT_HOST_GATE_AUTHORITY")
    need(authority["stage_SHA256"]==args.stage_sha and authority["source_manifest_SHA256"]==args.source_manifest_sha,
         "READ_ONLY_CHECK_SCOPE_PINS")
    report=inspect(authority,staged,args.source_manifest_sha,host=host_record)
    report["host_gate_SHA256"]=sha256(host).hexdigest()
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    return 0 if report["passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
