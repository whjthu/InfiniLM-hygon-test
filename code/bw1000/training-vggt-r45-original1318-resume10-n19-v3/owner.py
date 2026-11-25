"""Run eight-pair VGGT training and one fresh-process resume in one isolated Slurm step."""

import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import re
import stat
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time


R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = R / "results"
LABEL = "training-vggt-r45-original1318-resume10-n19-v3"
PREDECESSOR = "training-vggt-tum-eight-pairs25-recovery-v1"
PARENT = "1103606"
CPU, MEMORY = 4, 65536
FRAMEWORK = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
FRAMEWORK_MANIFEST = FRAMEWORK.parent / "source/FRAMEWORK_FILES.json"
FRAMEWORK_SHA = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
UPSTREAM = RESULTS / "training-vggt-native-provider-HIP-CPU-v1/upstream"
WEIGHTS = RESULTS / "training-vggt-original-public-model-v1/model"
WEIGHT_SHA = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
WEIGHT_STAT = (5026874952, 20701942394, 1794489337000000000)
BUNDLE_SHA = 'fdc6b68196db7403a00dada66bda274d08d737e83336cb4c3758362f7438b14c'
BACKEND_REL = "generation/src/gencuda/backends/vggt_training.py"
OVERLAY_SHA = '762fc0d74a6ebb3ebbd052a8bdfbe302477a518a6868ec322bd89ef07e16c998'
HOST_PY = "/usr/local/python3.12/bin/python3"
COLS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pinned(path, digest):
    require(path.is_file() and not path.is_symlink(), "missing or aliased input: " + path.name)
    raw = path.read_bytes()
    require(sha(raw) == digest, "input digest differs: " + path.name)
    return raw


def save(directory, name, data):
    with (directory / name).open("x", encoding="utf8") as out:
        json.dump(data, out, indent=2, sort_keys=True, allow_nan=False)


def weight_identity(path):
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and (value.st_size, value.st_ino, value.st_mtime_ns) == WEIGHT_STAT,
            "qualified model file identity changed")


def accounting(label, deadline=None, job=PARENT):
    remaining = 10 if deadline is None else min(10, deadline - time.time())
    if remaining <= 0:
        raise TimeoutError("original accounting deadline exhausted")
    output = subprocess.check_output(["sacct", "-n", "-P", "-j", job, "-o", COLS], timeout=remaining, text=True)
    rows = [row for line in output.splitlines() if len(row := line.split("|")) == 7 and row[1] == label]
    require(len(rows) <= 1, "duplicate exact Slurm step label")
    return rows[0] if rows else None


def phase_budget(phase, deadline, now):
    require(phase in ("pre", "gpu", "post"), "unknown step phase")
    require(math.isfinite(deadline) and math.isfinite(now), "invalid original clock")
    post_reserve = 90 if phase != "post" else 0
    final_write_reserve = 2
    terminal_seconds = 22
    launch_seconds = 5
    client_cap = 1205 if phase == "gpu" else 65
    available = math.floor(deadline - now) - post_reserve - final_write_reserve
    # Slurm rounds fractional minutes upward. Request only whole minutes that fit.
    slurm_seconds = 60 * math.floor(min(client_cap - launch_seconds,
                                      available - terminal_seconds - launch_seconds) / 60)
    require(slurm_seconds >= 60, "original epoch has no complete step and cleanup budget")
    client_seconds = slurm_seconds + launch_seconds
    return dict(slurm_seconds=slurm_seconds, client_seconds=client_seconds,
                slurm_time=f"{slurm_seconds // 3600:02d}:{slurm_seconds // 60 % 60:02d}:00",
                terminal_seconds=terminal_seconds, post_reserve_seconds=post_reserve,
                latest_terminal_wall=deadline-post_reserve-final_write_reserve)


from contextlib import ExitStack
import fcntl
from launch import container_command
from resume_bindings import REQUEST, OLD, CP_BYTES, FORENSIC, NODE, ORIGINAL_NODE, PLACEMENT_HISTORY_SHA, PRIOR_N19_ROWS, N19_PCIS, PREVIOUS_FAILED_LABEL, PREVIOUS_FAILURE_SHA, PREVIOUS_V2_FAILED_LABEL, PREVIOUS_V2_FAILURE_SHA, reference


PILOT_MOUNT=Path("/pilot")
REFERENCE_MOUNT=Path("/reference")
UPSTREAM_MANIFEST_SHA="64380b341ce377fb7e4b24c3f4f5986b6557b5339f4f7db57950dfa7e66b04c0"
BOUND_OVERLAY={
    "tum_vggt_pair.py":"c453a9c561c6411b8c2f25652f5c004c928db423c03ca0a45eba7d47ed9e3962",
    "domain_overlay.py":"6aef14e2326162edeca285a691e5629e74ca3800cb134348c668e2ff35f88816",
    "data_contract.py":"aef0765500140c237fb8c57adfa53d0ba024064c67a1716e9610ca3448f4d898",
}


def stage_ready(root, *, inside=False, report=None):
    from data_contract import verify_bundle

    def mark(phase):
        if report is not None:
            report["phase"]=phase
    original=REFERENCE_MOUNT if inside else OLD/"arms/continuous"
    base=PILOT_MOUNT if inside else OLD
    mark("ORIGINAL_CONTINUOUS_REFERENCE")
    record=reference(original)
    mark("ORIGINAL_EIGHT_TRAIN_PAIRS")
    require((base/"data").resolve(strict=True)==base/"data" and (base/"data").is_dir(),"original data layout")
    verify_bundle(base/"data",BUNDLE_SHA)
    require(sum(file.is_file() for file in (base/"data").rglob("*"))==65,"original 65 data members")
    mark("ORIGINAL_BACKEND_AND_THREE_DEPLOYED_OVERLAYS")
    framework=base/"framework"
    require(framework.resolve(strict=True)==framework and framework.is_dir(),"canonical original framework mount")
    pinned(framework/BACKEND_REL,OVERLAY_SHA)
    folder="generation/src/gencuda/"
    for name,digest in BOUND_OVERLAY.items():
        pinned(root/"src"/name,digest)
        pinned(framework/(folder+name),digest)
    if not inside:
        mark("ORIGINAL_393_PLUS_THREE_FRAMEWORK_SOURCES")
        files=json.loads(pinned(FRAMEWORK_MANIFEST,FRAMEWORK_SHA))["files"]
        expected={name:spec["SHA256"] for name,spec in files.items()}
        require(len(expected)==393 and BACKEND_REL in expected
                and not set(expected)&{folder+name for name in BOUND_OVERLAY},
                "original 393 framework base inventory")
        expected[BACKEND_REL]=OVERLAY_SHA
        expected.update({folder+name:digest for name,digest in BOUND_OVERLAY.items()})
        require(len(expected)==396,"original 396 framework source members")
        actual_python=set()
        non_source_pyc=0
        for path in framework.rglob("*"):
            if path.suffix==".pyc" and path.is_file():
                non_source_pyc+=1
            if path.suffix==".py" and path.is_file():
                require(path.resolve(strict=True)==path and not path.is_symlink(),"no aliased deployed Python source")
                actual_python.add(path.relative_to(framework).as_posix())
        require(actual_python=={name for name in expected if name.endswith(".py")},
                "original framework Python source inventory")
        for k,digest in expected.items():
            pinned(framework/k,digest)
        if report is not None:
            report.update(framework_pinned_sources=len(expected),framework_python_files=len(actual_python),
                          framework_pyc_excluded_from_source_denominator=non_source_pyc)
        mark("ORIGINAL_PINNED_VGGT_UPSTREAM")
        upstream=json.loads(pinned(root/"src/UPSTREAM_FILES.json",UPSTREAM_MANIFEST_SHA))["files"]
        for item in upstream:
            pinned(UPSTREAM/item["name"],item["SHA256"])
        if report is not None:
            report["upstream_pinned_files"]=len(upstream)
    mark("ORIGINAL_CP10_EXTENT")
    cp=original/"train/checkpoint-10.pt"
    require(cp.resolve(strict=True)==cp and cp.stat().st_size==CP_BYTES,"original exact checkpoint extent")
    mark("COMPLETE")
    return record


def fields(line):
    return dict(i.split("=",1) for i in line.split() if "=" in i)


def query(*args):
    return subprocess.check_output(args,text=True,timeout=6).splitlines()


def dual_clear(authority, report=None):
    """Check the current controller and queue; perform no scheduler mutation."""
    def mark(phase):
        if report is not None:
            report["phase"] = phase

    def nodes(spec):
        require(isinstance(spec, str) and 0 < len(spec) <= 8192, "unparsed scheduler placement")
        if re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}", spec):
            return {spec}
        require(re.fullmatch(r"f[0-9rnf,\[\]-]+", spec) is not None, "unparsed scheduler placement")
        out = query("scontrol", "show", "hostnames", spec)
        require(0 < len(out) <= 640 and len(set(out)) == len(out)
                and all(re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}", node) for node in out),
                "invalid scheduler placement expansion")
        return set(out)

    mark("PARENT_CONTROLLER")
    row, = query("scontrol", "show", "job", "-o", PARENT)
    info = fields(row)
    require(info["JobId"] == PARENT and info["JobState"] == "RUNNING"
            and info["UserId"].endswith(f"({os.getuid()})") and NODE in nodes(info["NodeList"])
            and "gres/dcu=320" in info["AllocTRES"].split(","), "exact parent job admission")
    mark("PARENT_DEADLINE")
    from datetime import datetime, timezone
    until = authority["valid_through_epoch"]
    require(type(until) in (int, float) and math.isfinite(until), "invalid authorized epoch")
    end = datetime.fromisoformat(info["EndTime"])
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    require(end.timestamp() >= until, "parent expires before authorized epoch")

    mark("PARENT_QUEUE")
    idle, = query("squeue", "-h", "-j", PARENT, "-o", "%i|%T|%N")
    own, state, host = idle.split("|")
    require(own == PARENT and state == "RUNNING" and NODE in nodes(host), "independent parent queue identity")

    mark("NODE_RESOURCE_RECORD")
    matches = []
    resource_fields = {"Nodes", "CPU_IDs", "Mem"}
    for row in query("scontrol", "-d", "show", "job", PARENT):
        item = fields(row)
        if "CPU_IDs" not in item and "Nodes" not in item:
            continue
        require(resource_fields <= item.keys(), "partial parent node resource record")
        if NODE in nodes(item["Nodes"]):
            matches.append((item["CPU_IDs"], item["Mem"]))
    require(len(matches) == 1, "exact node parent CPU and memory extents")
    cpus, mem = matches[0]
    mark("NODE_CPU_MEMORY")
    cores = set()
    require(0 < len(cpus) <= 8192, "invalid CPU list length")
    for portion in cpus.split(","):
        require(re.fullmatch(r"[0-9]{1,5}(?:-[0-9]{1,5})?", portion) is not None, "invalid CPU list")
        ids = portion.split("-")
        lo, hi = int(ids[0]), int(ids[-1])
        require(lo <= hi < 16384, "invalid CPU interval")
        interval = set(range(lo, hi + 1))
        require(not cores.intersection(interval), "overlapping CPU intervals")
        cores.update(interval)
    require(re.fullmatch(r"[0-9]{1,12}", mem) is not None, "invalid parent node memory")
    require(len(cores) >= CPU and int(mem) >= MEMORY, "parent per-node CPU/memory insufficient")
    if report is not None:
        report.update(node_cpu_count=len(cores), node_mem_MiB=int(mem))

    mark("CONTROLLER_STEPS")
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
    for row in query("scontrol", "show", "step", PARENT, "-o"):
        item = fields(row)
        step = item["StepId"]
        if re.fullmatch(PARENT + r"\.[0-9]+", step) and item["State"].split()[0].rstrip("+") not in terminal:
            require(NODE not in nodes(item["NodeList"]), "live owner in controller, no reclaim")
    mark("QUEUE_STEPS")
    for row in query("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"):
        handle, host = row.split("|")
        if re.fullmatch(PARENT + r"\.[0-9]+", handle):
            require(NODE not in nodes(host), "live owner in queue, no reclaim")
    mark("COMPLETE")


GATE_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError, subprocess.SubprocessError)


def record_failure(report, error):
    report.update(passed=False, error_class=type(error).__name__, failure_code=report["phase"])
    if isinstance(error, OSError):
        report["errno"] = error.errno
    return report


def stage_check(root):
    report = dict(request=REQUEST, phase="START", passed=False, Slurm_or_GPU_invoked=False)
    try:
        stage_ready(root, report=report)
        report["phase"] = "BASE_WEIGHT_STAT"
        weight_identity(WEIGHTS / "model.pt")
        report.update(phase="COMPLETE", passed=True)
    except GATE_ERRORS as error:
        record_failure(report, error)
    return report


def current_admission(authority):
    report = dict(request=REQUEST, phase="START", passed=False, scheduler_read_only=True)
    try:
        dual_clear(authority, report)
        report["passed"] = True
    except GATE_ERRORS as error:
        record_failure(report, error)
    return report


def readonly(stage_root, until):
    stage = stage_check(stage_root)
    admission = current_admission({"valid_through_epoch": until}) if stage["passed"] else None
    result = dict(request=REQUEST, mode="READ_ONLY", stage=stage, admission=admission,
                  scheduler_mutations=0, GPU_steps=0, binds_no_owner=True, training_ready=False)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if admission and admission["passed"] else 1


def check_current_and_save(root, name, authority):
    report = current_admission(authority)
    save(root, name, report)
    require(report["passed"], "current scheduler admission failed")


def resources(row,phase):
    want=({"cpu=4","gres/dcu=1","mem=64G","node=1"} if phase=="gpu"
          else {"cpu=2","mem=2G","node=1"})
    alt=(want-({"mem=64G"} if phase=="gpu" else {"mem=2G"})) | (
         {"mem=65536M"} if phase=="gpu" else {"mem=2048M"})
    tokens=row[6].split(",")
    return len(tokens)==len(want) and set(tokens) in (want,alt)


def step(root,phase,payload,deadline,unlock=None):
    from slurm_accounting import wait_terminal,is_terminal
    from launch import srun_command
    label=LABEL+"-"+phase
    require(accounting(label,deadline) is None,"existing phase label, reconcile only")
    require((phase=="gpu")==bool(unlock),"GPU owner requires a lock handoff")
    budget=phase_budget(phase,deadline,time.time())
    argv=srun_command(phase,payload,budget["slurm_time"])
    save(root,phase+"-step-intent.json",dict(request=REQUEST,argv=argv,budget=budget))
    child=None
    rc=None
    known=None
    failure=None
    held={}
    try:
        with (root/(phase+".log")).open("xb") as log:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if unlock:
                end=min(time.time()+10,budget["latest_terminal_wall"])
                while time.time()<end:
                    row=accounting(label,end)
                    if row:
                        require(re.fullmatch(PARENT+r"\.[0-9]+",row[0]) and row[5]==NODE and resources(row,phase),
                                "inexact numeric GPU handoff")
                        if row[2]=="RUNNING" or is_terminal(row):
                            known=row[0]
                            save(root,"gpu-shared-lock-handoff.json",dict(request=REQUEST,step=known,tres=row[6]))
                            unlock()
                            break
                    time.sleep(.25)
                require(known is not None,"unknown GPU handoff, no replay")
            rc=child.wait(timeout=min(budget["client_seconds"],max(1,budget["latest_terminal_wall"]-time.time())))
    except (OSError,ValueError,TimeoutError,subprocess.SubprocessError) as error:
        failure=type(error).__name__
    finally:
        from held_child import cleanup_held
        anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True) if child is not None else {}
        try:
            held=cleanup_held(child,anchor,min(time.time()+12,budget["latest_terminal_wall"]))
        finally:
            if unlock and known is None:
                save(root,"gpu-handoff-unknown.json",dict(request=REQUEST,ordinary_exclusion_retained=True,no_replay=True))
                unlock()
    until=min(time.time()+budget["terminal_seconds"],budget["latest_terminal_wall"])
    result=wait_terminal(lambda:accounting(label,until) if time.time()<until else None,
                         time.monotonic()+max(0,until-time.time()),max_observations=9)
    row=result["row"]
    valid=bool(result["terminal_visible"] and row and re.fullmatch(PARENT+r"\.[0-9]+",row[0])
               and row[5]==NODE and resources(row,phase) and (known is None or known==row[0]))
    result.update(rc=rc,terminal_visible=valid,error_class=failure,held=held,
                  passed=bool(valid and row[2:4]==["COMPLETED","0:0"] and rc==0 and failure is None))
    save(root,phase+"-accounting.json",result)
    return result


def controller():
    os.umask(0o077)
    root=RESULTS/LABEL
    intent=json.loads((root/"intent.json").read_bytes())
    epoch,deadline=intent["epoch"],intent["deadline"]
    require(deadline==epoch+1800 and epoch<=time.time()<epoch+10 and intent["request"]==REQUEST,"exact new epoch")
    manifest=json.loads(pinned(root/"SOURCE_FILES.json",intent["manifest_SHA256"]))
    for name,item in manifest["files"].items():
        target=root/"src"/name
        require(target.resolve(strict=True)==target and len(pinned(target,item["SHA256"]))==item["bytes"],"exact source")
    claim=json.loads((root/"src/CLAIM.json").read_bytes())
    require(claim["GPU_GO"] is True and claim["request"]==REQUEST and claim["label"]==LABEL
            and claim["allow_continuous_replay"] is False,"exact new scope authorization")
    authority=json.loads(pinned(root/"authority.json",claim["fresh_root_authority_SHA256"]))
    require(authority["request"]==REQUEST and authority["label"]==LABEL and authority["parent"]==PARENT
            and authority["node"]==NODE and authority["original_node"]==ORIGINAL_NODE
            and authority["placement_history_SHA256"]==PLACEMENT_HISTORY_SHA
            and authority["previous_failed_scope_SHA256"]==PREVIOUS_FAILURE_SHA
            and authority["previous_v2_failed_scope_SHA256"]==PREVIOUS_V2_FAILURE_SHA
            and authority["root_sole_dispatcher"] is True
            and authority["ordinary_exclusion_until_independent_post"] is True and authority["approved_one_DCU_once"] is True
            and authority["original_live_lineage_SHA256"]==FORENSIC and authority["prior_node_unknown_conflict"] is False
            and 0<=epoch-authority["observed_epoch"]<=90 and deadline<=authority["valid_through_epoch"],
            "independent current root authority")
    stage_report = stage_check(root)
    save(root, "stage-pre-scheduler.json", stage_report)
    require(stage_report["passed"], "failed versioned stage before any Slurm/GPU gate")
    history = dict(request=REQUEST, phase="PRIOR_NO_GPU_LABELS", passed=False, scopes_checked=0)
    try:
        for previous in (PREVIOUS_FAILED_LABEL, PREVIOUS_V2_FAILED_LABEL):
            require(all(accounting(previous+"-"+p, deadline) is None for p in ("pre", "gpu", "post")),
                    "original no-GPU failure now has an accounting row, reconcile only")
            history["scopes_checked"] += 1
        history["phase"] = "ORIGINAL_1318_ACCOUNTING"
        for phase, expected in (("pre", "1103606.1317"), ("gpu", "1103606.1318"), ("post", "1103606.1319")):
            row = accounting(OLD.name+"-"+phase, deadline)
            require(row and row[0] == expected and row[2:4] == ["COMPLETED", "0:0"]
                    and row[5] == ORIGINAL_NODE and resources(row, phase), "original .1318 actual accounting")
        history["phase"] = "ORIGINAL_N19_ACCOUNTING"
        for historical in PRIOR_N19_ROWS:
            require(accounting(historical[1], deadline) == historical, "historical .3204/.3205 C0 n19 binding")
        history.update(phase="COMPLETE", passed=True)
    except GATE_ERRORS as exc:
        record_failure(history, exc)
    save(root, "history-pre-scheduler.json", history)
    require(history["passed"], "historical accounting must remain exact")

    work=root/"src/owner.py"
    snapshot_cmd=[HOST_PY,"-B",str(work),"snapshot"]
    stages={}
    error=None
    before=after=None
    checkpoint="SHARED_LOCK"
    failure_code=None
    try:
        with ExitStack() as lock_stack:
            lock=lock_stack.enter_context((RESULTS/".training80-dcu-standby.lock").open("rb"))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            checkpoint="ADMISSION_BEFORE_PRE"
            check_current_and_save(root,"admission-before-pre.json",authority)
            checkpoint="NEW_EXACT_LABELS"
            require(all(accounting(LABEL+"-"+phase,deadline) is None for phase in ("pre","gpu","post")),"existing owner")
            checkpoint="CUSTODY_INTENT"
            save(root,"owner-custody-intent.json",dict(request=REQUEST,epoch=epoch))
            checkpoint="PHYSICAL_PRE_STEP"
            stages["pre"]=step(root,"pre",snapshot_cmd,deadline)
            require(stages["pre"]["passed"],"fresh physical CPU pre required")
            checkpoint="PHYSICAL_PRE_IDENTITY"
            before=json.loads((root/"pre.log").read_bytes())
            require(before["step"]==stages["pre"]["row"][0] and before["node"]==NODE
                    and before["pci"]==list(N19_PCIS) and before["kfd_holder_count"]==0
                    and before["actual_host_numeric_cgroup"] is True,"read actual fresh n19 pre")
            save(root,"before.json",before)
            checkpoint="ADMISSION_AFTER_PRE"
            check_current_and_save(root,"admission-after-pre.json",authority)
            checkpoint="GPU_STEP"
            stages["gpu"]=step(root,"gpu",container_command(root),deadline,lock_stack.close)
    except (OSError,ValueError,TypeError,KeyError,subprocess.SubprocessError) as exc:
        error=type(exc).__name__
        failure_code=checkpoint
    if stages.get("gpu",{}).get("terminal_visible"):
        checkpoint="INDEPENDENT_POST_STEP"
        try:
            stages["post"]=step(root,"post",snapshot_cmd,deadline)
            if stages["post"]["passed"]:
                checkpoint="INDEPENDENT_POST_IDENTITY"
                after=json.loads((root/"post.log").read_bytes())
                require(after["boot_id"]==before["boot_id"] and after["pci"]==before["pci"]
                        and after["kfd_holder_count"]==0,"actual original independent post release")
                save(root,"after.json",after)
        except (OSError,ValueError,TypeError,KeyError,subprocess.SubprocessError) as exc:
            error=type(exc).__name__
            failure_code=checkpoint
            after=None
    native_proof=root/"native-result.json"
    result=dict(request=REQUEST,epoch=epoch,deadline=deadline,error_class=error,failure_code=failure_code,stages=stages,
                release_candidate=after is not None,planned_updates=15,continued_replay=False,
                original_1318_replayed=False,formal_performance=False,
                training_ready=False,native_result_SHA256=sha(native_proof.read_bytes()) if native_proof.is_file() else None)
    save(root,"outcome.json",result)
    print(json.dumps(result,sort_keys=True),flush=True)
    return 0 if all(stages.get(k,{}).get("passed") for k in ("pre","gpu","post")) and native_proof.is_file() else 1


def canonical_pci(value):
    value=value.lower()
    if len(value)==7:
        value="0000:"+value
    require(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",value),"invalid device PCI identity")
    domain,bus=value.split(":",1)
    return f"{int(domain,16):04x}:{bus}"


def normalize(entries):
    pcis=sorted(canonical_pci(v) for v in entries)
    require(len(pcis)==len(set(pcis))==8,"exact eight same-node PCI identities")
    return pcis


def host_snapshot(root):
    import snapshot

    snapshot.NODE = NODE
    snapshot.STEP = __import__("re").compile(PARENT + r"\.\d+")
    numeric=os.environ.get("SLURM_STEP_ID","")
    require(os.environ.get("SLURM_JOB_ID")==PARENT and re.fullmatch(r"[0-9]+",numeric)
            and socket.gethostname().split(".")[0]==NODE,"current n19 numeric CPU step")
    cgroups=Path("/proc/self/cgroup").read_text("ascii").splitlines()
    require(any({"job_"+PARENT,"step_"+numeric}<=set(item.split(":",2)[-1].split("/")) for item in cgroups),
            "same actual current CPU job step cgroup")
    row = snapshot.snapshot(False)
    row["pci"] = normalize(row["pci"])
    require(row["step"]==PARENT+"."+numeric,"same CPU step")
    row["actual_host_numeric_cgroup"]=True
    require(row["kfd_holder_count"] == 0 and row["full_fd_inventory_performed"], "node still holds KFD")
    owner = root / "assignment.json"
    if owner.is_file():
        assigned = json.loads(owner.read_bytes())
        try:
            fields = Path(f"/proc/{assigned['pid']}/stat").read_text().rsplit(") ", 1)[1].split()
            require(fields[0] == "Z" or int(fields[19]) != assigned["startticks"], "native process still live")
        except FileNotFoundError:
            pass
    print(json.dumps(row), flush=True)


def native(root):
    import snapshot
    from assigned_kfd import assigned_kfd
    import re

    before = json.loads((root / "before.json").read_bytes())
    stage_ready(Path("/pilot"),inside=True)
    intent = json.loads((root / "intent.json").read_bytes())
    require(time.time() < intent["deadline"] - 45 and os.environ["SLURM_JOB_ID"] == PARENT
            and socket.gethostname().split(".")[0] == NODE and len(os.sched_getaffinity(0)) >= CPU,
            "wrong native allocation")
    require(sorted(canonical_pci(p) for p in snapshot.drm_inventory()) == before["pci"] and not snapshot.kernel_holders()
            and before["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            and 0 <= time.monotonic_ns() - before["monotonic_ns"] <= 45_000_000_000, "stale or occupied device snapshot")
    numeric=os.environ.get("SLURM_STEP_ID","")
    require(re.fullmatch(r"[0-9]+",numeric) and numeric not in ("1317","1318","1319")
            and any({"job_"+PARENT,"step_"+numeric}<=set(row.split(":",2)[-1].split("/"))
                    for row in Path("/proc/self/cgroup").read_text().splitlines()),"exact numeric Slurm GPU cgroup")
    hip = ctypes.CDLL("libamdhip64.so")
    count, pci_raw = ctypes.c_int(), ctypes.create_string_buffer(32)
    require(hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1
            and hip.hipDeviceGetPCIBusId(pci_raw, len(pci_raw), 0) == 0, "one HIP device required")
    pci = canonical_pci(pci_raw.value.decode())
    require(pci in before["pci"], "HIP PCI does not belong to the node")
    save(root, "assignment.json", dict(step=PARENT+"."+os.environ["SLURM_STEP_ID"], node=NODE, pci=pci,
                                        kfd=assigned_kfd(pci), pid=os.getpid(), startticks=int(
                                            Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19])))
    require(importlib.metadata.version("torch") ==
            "2.10.0+das.opt1.dtk2604.2606161421.g91a580", "qualified Torch provider differs")
    env = dict(os.environ, VGGT_DOMAIN_EPOCH=str(intent["epoch"]),
               VGGT_DOMAIN_DEADLINE=str(intent["deadline"] - 100))
    receipts = {}
    for arm in ("resume10",):
        require(time.time() < intent["deadline"] - 125, "original training budget exhausted")
        args = [sys.executable, "-B", "-u", "/pilot/src/native_worker.py", "--arm", arm,
                "--data", "/pilot/data", "--weights", "/weights", "--upstream", "/vggt-source",
                "--output-root", "/pilot/arms", "--bundle-sha256", BUNDLE_SHA, "--overlay-sha256", OVERLAY_SHA]
        with (root / (arm + ".log")).open("xb") as log:
            rc = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, env=env,
                                timeout=max(1, int(intent["deadline"] - time.time()) - 110)).returncode
        require(rc == 0, "native training arm failed: " + arm)
        proof = root / "arms" / arm / "receipt.json"
        receipts[arm] = dict(SHA256=sha(proof.read_bytes()), value=json.loads(proof.read_bytes()))
        weight_identity(Path("/weights/model.pt"))
    from resume_bindings import reference, compare
    comparison=compare(reference(Path("/reference")),receipts["resume10"]["value"])
    save(root,"native-result.json",dict(receipts=receipts,comparison=comparison,actual_DCU=1))
    require(comparison["comparison_passed"],"resume10 differs from independently retained .1318")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "read-only-admission":
        stage_root = Path(sys.argv[2])
        require(stage_root.is_absolute() and stage_root.resolve(strict=True) == stage_root,
                "canonical prior source stage path required")
        raise SystemExit(readonly(stage_root, float(sys.argv[3])))
    if sys.argv[1:] == ["snapshot"]:
        host_snapshot(Path(__file__).resolve().parents[1])
    elif sys.argv[1:] == ["native"]:
        native(Path("/pilot"))
    elif sys.argv[1:]:
        raise SystemExit("unrecognized owner arguments")
    else:
        raise SystemExit(controller())
