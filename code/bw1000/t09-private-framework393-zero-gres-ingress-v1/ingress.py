"""One no-model, no-SDK production-image ingress proof for the exact private T09 tree."""
import argparse
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
from private_tree import DENOMINATOR, MANIFEST, REQUEST, SHARED, inspect_tree, validate_manifest, need


BACKEND_SHA="2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae"
LABEL="t09-private-framework393-ingress-3d616fad903f"
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


def verify_readonly_mount():
    raw=Path("/proc/self/mountinfo").read_text("ascii")
    exact=[]
    for line in raw.splitlines():
        columns=line.split()
        if len(columns)>=10 and columns[4]==str(SHARED):
            exact.append(columns)
    need(len(exact)==1 and "ro" in exact[0][5].split(",")
         and (os.statvfs(SHARED).f_flag & os.ST_RDONLY)!=0,"EXACT_PRODUCTION_FRAMEWORK_READONLY_OVERLAY")
    return True


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


def inspect(authority, stage, source_manifest, *, root=SHARED, manifest=MANIFEST, step_check=establish_step,
            mount_check=verify_readonly_mount, verify=verify_framework, providers=metadata_providers):
    report=dict(request=REQUEST,phase="STEP",passed=False,planned=DENOMINATOR,before_verified=0,
                after_verified=0,providers_planned=1+len(REQUIREMENTS),providers_verified=0,
                model_loaded=False,weight_payload_bytes_read=0,GPU_SDK_calls=0,optimizer_updates=0,formal_performance=False)
    try:
        report["step"]=step_check(authority)
        need(sys.dont_write_bytecode is True and os.environ.get("PYTHONDONTWRITEBYTECODE")=="1"
             and sys.pycache_prefix is None,"NO_PYTHON_BYTECODE_CACHE_READ_PATH_CHANGE_OR_WRITES")
        report["phase"]="MOUNT"
        need(mount_check() is True,"READ_ONLY_FRAMEWORK_MOUNT")
        need(stage["request"]==REQUEST and stage["passed"] is True and stage["verified_original"]==DENOMINATOR
             and stage["manifest_SHA256"]==MANIFEST_SHA256,"EXACT_NEW_PRIVATE_STAGE_RESULT")
        pins=validate_manifest(manifest.read_bytes())
        report["phase"]="ORIGINAL_BEFORE"
        original_before=verify(root,manifest)
        before=inspect_tree(root,pins)
        report.update(before_verified=original_before["framework_files"],before_inventory=before)
        report["phase"]="PROVIDERS"
        report["providers"]=providers(root,pins,sys.path)
        report["providers_verified"]=len(report["providers"])
        report["phase"]="ORIGINAL_AFTER"
        original_after=verify(root,manifest)
        after=inspect_tree(root,pins)
        report.update(after_verified=original_after["framework_files"],after_inventory=after)
        need(before==after==stage["staged_inventory"] and original_after==original_before==stage["original_verifier"]
             and report["before_verified"]==report["after_verified"]==DENOMINATOR,"EXACT_UNCHANGED_FRAMEWORK_INVENTORY")
        need(not any(name=="torch" or name.startswith("torch.") or name=="gencuda" or name.startswith("gencuda.")
                     for name in sys.modules),"NO_ACTUAL_TORCH_OR_GPU_PROVIDER_IMPORTED")
        report.update(phase="COMPLETE",passed=True,bound_source_manifest_SHA256=source_manifest,readonly_overlay=True)
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
    staged=pinned(scope/"stage-result.json",args.stage_sha)
    need(authority["stage_SHA256"]==args.stage_sha and authority["source_manifest_SHA256"]==args.source_manifest_sha,
         "READ_ONLY_CHECK_SCOPE_PINS")
    report=inspect(authority,staged,args.source_manifest_sha)
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    return 0 if report["passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
