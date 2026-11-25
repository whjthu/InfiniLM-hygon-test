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
LABEL = "training-vggt-r45-original1318-resume10-n19-v1"
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
from resume_bindings import REQUEST, OLD, CP_BYTES, FORENSIC, NODE, ORIGINAL_NODE, PLACEMENT_HISTORY_SHA, PRIOR_N19_ROWS, N19_PCIS, reference


def stage_ready(root, *, inside=False):
    from data_contract import verify_bundle

    record=reference(Path("/reference") if inside else OLD/"arms/continuous")
    base=Path("/pilot") if inside else OLD
    require((base/"data").is_dir(),"original data layout")
    verify_bundle(base/"data",BUNDLE_SHA)
    require(sum(file.is_file() for file in (base/"data").rglob("*"))==65,"original 65 data members")
    pinned(base/"framework"/BACKEND_REL,OVERLAY_SHA)
    if not inside:
        files=json.loads(pinned(FRAMEWORK_MANIFEST,FRAMEWORK_SHA))["files"]
        expected={name:spec["SHA256"] for name,spec in files.items()}
        expected[BACKEND_REL]=OVERLAY_SHA
        folder="generation/src/gencuda/"
        for name in ("tum_vggt_pair.py","domain_overlay.py","data_contract.py"):
            expected[folder+name]=sha((root/"src"/name).read_bytes())
        require(len(expected)==396,"original 396 framework source members")
        require({file.relative_to(base/"framework").as_posix() for file in (base/"framework").rglob("*.py")}==
                {name for name in expected if name.endswith(".py")},"original framework Python source inventory")
        for k,digest in expected.items():
            pinned(base/"framework"/k,digest)
        for item in json.loads((root/"src/UPSTREAM_FILES.json").read_bytes())["files"]:
            pinned(UPSTREAM/item["name"],item["SHA256"])
    cp=(Path("/reference") if inside else OLD/"arms/continuous")/"train/checkpoint-10.pt"
    require(cp.resolve(strict=True)==cp and cp.stat().st_size==CP_BYTES,"original exact checkpoint extent")
    return record


def fields(line):
    return dict(i.split("=",1) for i in line.split() if "=" in i)


def query(*args):
    return subprocess.check_output(args,text=True,timeout=6).splitlines()


def dual_clear(authority):
    def nodes(spec):
        if re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",spec):
            return {spec}
        require(bool(spec) and spec not in ("(null)","Unknown"),"unparsed scheduler placement")
        out=set(query("scontrol","show","hostnames",spec))
        require(bool(out),"empty scheduler placement")
        return out
    row,=query("scontrol","show","job","-o",PARENT)
    info=fields(row)
    require(info["JobId"]==PARENT and info["JobState"]=="RUNNING" and info["UserId"].endswith(f"({os.getuid()})")
            and NODE in nodes(info["NodeList"]) and "gres/dcu=320" in info["AllocTRES"].split(","),
            "exact parent job admission")
    from datetime import datetime, timezone
    limit=datetime.fromisoformat(info["EndTime"]).replace(tzinfo=timezone.utc).timestamp()
    require(limit>=authority["valid_through_epoch"],"parent expires before authorized epoch")
    idle,=query("squeue","-h","-j",PARENT,"-o","%i|%T|%N")
    own,state,host=idle.split("|")
    require(own==PARENT and state=="RUNNING" and NODE in nodes(host),"independent parent queue identity")
    matches=[]
    for row in query("scontrol","-d","show","job",PARENT):
        for n,cpus,mem in re.findall(r"Nodes=(S+)s+CPU_IDs=(S+)s+Mem=(S+)",row):
            if NODE in nodes(n):
                matches.append((cpus,mem))
    require(len(matches)==1,"exact node parent CPU and memory extents")
    cpus,mem=matches[0]
    portions=[p.split("-") for p in cpus.split(",")]
    require(all(1<=len(p)<=2 and all(x.isdecimal() for x in p) for p in portions),"exact CPU list")
    require(sum(int(p[-1])-int(p[0])+1 for p in portions)>=4 and mem.isdecimal() and int(mem)>=65536,
            "parent per-node CPU/memory insufficient")
    terminal={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL"}
    for row in query("scontrol","show","step",PARENT,"-o"):
        item=fields(row)
        step=item["StepId"]
        if re.fullmatch(PARENT+r"\.[0-9]+",step) and item["State"].split()[0].rstrip("+") not in terminal:
            require(NODE not in nodes(item["NodeList"]),"live owner in controller, no reclaim")
    for row in query("squeue","--steps","-h","-j",PARENT,"-o","%i|%N"):
        handle,host=row.split("|")
        if re.fullmatch(PARENT+r"\.[0-9]+",handle):
            require(NODE not in nodes(host),"live owner in queue, no reclaim")


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
            and authority["root_sole_dispatcher"] is True
            and authority["ordinary_exclusion_until_independent_post"] is True and authority["approved_one_DCU_once"] is True
            and authority["original_live_lineage_SHA256"]==FORENSIC and authority["prior_node_unknown_conflict"] is False
            and 0<=epoch-authority["observed_epoch"]<=90 and deadline<=authority["valid_through_epoch"],
            "independent current root authority")
    stage_ready(root)
    weight_identity(WEIGHTS/"model.pt")
    for phase,expected in (("pre","1103606.1317"),("gpu","1103606.1318"),("post","1103606.1319")):
        row=accounting(OLD.name+"-"+phase,deadline)
        require(row and row[0]==expected and row[2:4]==["COMPLETED","0:0"] and row[5]==ORIGINAL_NODE and resources(row,phase),
                "original .1318 actual accounting")
    for historical in PRIOR_N19_ROWS:
        require(accounting(historical[1],deadline)==historical,"historical .3204/.3205 C0 n19 binding")
    work=root/"src/owner.py"
    snapshot_cmd=[HOST_PY,"-B",str(work),"snapshot"]
    stages={}
    error=None
    before=after=None
    try:
        with ExitStack() as lock_stack:
            lock=lock_stack.enter_context((RESULTS/".training80-dcu-standby.lock").open("rb"))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            dual_clear(authority)
            require(all(accounting(LABEL+"-"+phase,deadline) is None for phase in ("pre","gpu","post")),"existing owner")
            save(root,"owner-custody-intent.json",dict(request=REQUEST,epoch=epoch))
            stages["pre"]=step(root,"pre",snapshot_cmd,deadline)
            require(stages["pre"]["passed"],"fresh physical CPU pre required")
            before=json.loads((root/"pre.log").read_bytes())
            require(before["step"]==stages["pre"]["row"][0] and before["node"]==NODE
                    and before["pci"]==list(N19_PCIS) and before["kfd_holder_count"]==0
                    and before["actual_host_numeric_cgroup"] is True,"read actual fresh n19 pre")
            save(root,"before.json",before)
            dual_clear(authority)
            stages["gpu"]=step(root,"gpu",container_command(root),deadline,lock_stack.close)
    except (OSError,ValueError,TypeError,KeyError,subprocess.SubprocessError) as exc:
        error=type(exc).__name__
    if stages.get("gpu",{}).get("terminal_visible"):
        try:
            stages["post"]=step(root,"post",snapshot_cmd,deadline)
            if stages["post"]["passed"]:
                after=json.loads((root/"post.log").read_bytes())
                require(after["boot_id"]==before["boot_id"] and after["pci"]==before["pci"]
                        and after["kfd_holder_count"]==0,"actual original independent post release")
                save(root,"after.json",after)
        except (OSError,ValueError,TypeError,KeyError,subprocess.SubprocessError) as exc:
            error=type(exc).__name__
            after=None
    native_proof=root/"native-result.json"
    result=dict(request=REQUEST,epoch=epoch,deadline=deadline,error_class=error,stages=stages,
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


if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)==2 else "controller"
    if mode=="snapshot":
        host_snapshot(Path(__file__).resolve().parents[1])
    elif mode=="native":
        native(Path("/pilot"))
    else:
        raise SystemExit(controller())
