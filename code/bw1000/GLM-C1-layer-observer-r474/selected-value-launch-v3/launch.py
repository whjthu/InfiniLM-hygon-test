"""Execute one frozen selected-value component scope in the I02 parent."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from slurm_accounting import wait_terminal
from order_effect import summarize as summarize_order

PARENT = "1103606"
NODES = ("f13r1n05", "f13r1n07")
LABEL = "GLM474-selected-value-GPU-v3"
SRC = Path(__file__).resolve().parent
OUT = SRC.parent
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList"
PHASES = ("before-create", "after-create", "numerics-passed", "before-close", "after-close")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(path, value):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def query(end, *args):
    left = end - time.monotonic()
    require(left > 0, "original launch budget exhausted")
    return subprocess.check_output(args, text=True, timeout=min(left, 8)).splitlines()


def nodes(end, expression):
    return set(query(end, "scontrol", "show", "hostnames", expression))


def account(end, label):
    rows = [line.split("|") for line in query(end, "sacct", "-n", "-P", "-j", PARENT,
             "--format="+COLUMNS) if "|"+label+"|" in line]
    require(len(rows) <= 1, "ambiguous accounting label")
    if not rows:
        return None
    row = rows[0]
    require(len(row)==6 and row[1]==label and re.fullmatch(PARENT+r"\.\d+", row[0]),
            "unexpected Slurm accounting row")
    return row


def preflight(end):
    parent = query(end, "squeue", "-h", "-j", PARENT, "-o", "%T|%D|%N")
    require(len(parent)==1 and parent[0].split("|")[:2]==["RUNNING","40"],
            "I02 parent allocation changed")
    require(set(NODES) <= nodes(end,parent[0].split("|")[2]), "I02 nodes changed")
    controller = query(end, "scontrol", "show", "step", PARENT, "-o")
    require(controller, "controller returned no steps for the live allocation")
    for line in controller:
        fields = dict(word.split("=",1) for word in line.split() if "=" in word)
        step = fields.get("StepId","")
        require(re.fullmatch(PARENT+r"\.(?:\d+|batch|extern)",step), "unknown controller step")
        if not step.endswith((".batch",".extern")):
            hosts = fields.get("NodeList")
            require(hosts not in (None,"","(null)","None","(none)") and
                    set(NODES).isdisjoint(nodes(end,hosts)), "live or unresolved step overlaps I02")
    failed = account(end,"GLM474-selected-value-GPU-v1-gate")
    released = account(end,"GLM474-selected-value-GPU-v1-post")
    require(failed and failed[:1]==["1103606.126"] and failed[2:4]==["OUT_OF_MEMORY","0:125"]
            and released and released[:1]==["1103606.127"] and released[2:4]==["COMPLETED","0:0"]
            and nodes(end,failed[5])==nodes(end,released[5])==set(NODES),
            "original OOM and cleanup must be reconciled before the correction")
    original = account(end,"GLM474-selected-value-GPU-v2-gate")
    original_post = account(end,"GLM474-selected-value-GPU-v2-post")
    require(original and original[0]=="1103606.129" and original[2:4]==["COMPLETED","0:0"]
            and original_post and original_post[0]=="1103606.130" and original_post[2:4]==["COMPLETED","0:0"]
            and nodes(end,original[5])==nodes(end,original_post[5])==set(NODES),
            "prior component and cleanup must be reconciled")
    for phase in ("pre","gate","post"):
        require(account(end,LABEL+"-"+phase) is None, "same-scope Slurm step already exists")


def step(phase, end, stop, env):
    label = LABEL+"-"+phase
    gpu = phase=="gate"
    path = OUT/(phase+"-receipt.json")
    entry = dict(phase=phase, label=label, state="SUBMISSION_POSSIBLE", row=None,
                 accounting_columns=COLUMNS.split(","), observations=[], passed=False)
    with path.open("x") as stream:
        json.dump(entry,stream)
        stream.flush()
        os.fsync(stream.fileno())
    argv = ["srun","--jobid="+PARENT,"--job-name="+label,"--exact","--immediate=5",
            "--nodes=2","--ntasks=2","--ntasks-per-node=1","--distribution=block:block",
            "--cpu-bind=none","--nodelist="+",".join(NODES),"--kill-on-bad-exit=1",
            "--cpus-per-task="+("64" if gpu else "1"),"--mem="+("224000M" if gpu else "1024M"),
            "--gres="+("dcu:8" if gpu else "none"),"--time="+("5" if gpu else "1"),
            "/usr/local/python3.12/bin/python3","-B",str(SRC/"probe.py"),phase,str(OUT)]
    child = None
    try:
        require(stop>time.monotonic(),"original step budget exhausted")
        with (OUT/(phase+".log")).open("xb") as log:
            child = subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry["srun_PID"] = child.pid
            entry["srun_startticks"] = Path(f"/proc/{child.pid}/stat").read_text().rsplit(") ",1)[1].split()[19]
            save(path,entry)
            child.wait(timeout=max(.01,stop-time.monotonic()))
    except Exception as exc:
        entry["error"] = type(exc).__name__
    finally:
        try:
            if child and child.poll() is None:
                entry["forced_cleanup"] = True
                for sig in (signal.SIGTERM,signal.SIGKILL):
                    try:
                        os.killpg(child.pid,sig)
                    except ProcessLookupError:
                        break
                    try:
                        child.wait(timeout=4)
                        break
                    except subprocess.TimeoutExpired:
                        entry["forced_cleanup"] = True
            entry["returncode"] = child.poll() if child else None
            def observe():
                row = account(end,label)
                entry["observations"].append(row)
                save(path,entry)
                return row
            result = wait_terminal(observe,min(end,time.monotonic()+20),interval=.5,max_observations=20)
            row = result["row"]
            if result["terminal_visible"]:
                entry.update(row=row,state="TERMINAL")
                entry["passed"] = bool(row[2:4]==["COMPLETED","0:0"] and entry["returncode"]==0
                                       and "error" not in entry and not entry.get("forced_cleanup")
                                       and nodes(end,row[5])==set(NODES))
            else:
                entry["state"] = "UNKNOWN"
        except Exception as exc:
            entry["accounting_error"] = type(exc).__name__
        finally:
            save(path,entry)
    return entry


def snapshots(phase, receipt):
    require(receipt["passed"], f"{phase} Slurm step failed")
    result = {}
    for node in NODES:
        item = json.loads((OUT/phase/(node+".json")).read_bytes())
        require(item["node"]==node and item["step"]==receipt["row"][0] and
                item["kfd_holder_count"]==0 and item["port_bindable"] is True and
                len(set(item["pci"]))==8, "hardware probe evidence does not match accounting")
        result[node] = item
    return result


def audit_owners(gpu, gate, claim, epoch):
    identity = dict(step=gpu["row"][0],claim=claim,epoch=epoch)
    phases = None
    report = {}
    cards = 0
    uids = None
    for node, name in enumerate(NODES):
        leader = json.loads((OUT/f"leader-{node}.json").read_bytes())
        require(leader["identity"]==identity and leader["node"]==node and leader["passed"]
                and leader["exitcodes"]==[0]*8 and len(leader["owners"])==8
                and [x["phase"] for x in leader["phases"]]==list(PHASES), "leader incomplete")
        require(phases in (None,leader["phases"]), "leader gate digests disagree")
        phases = leader["phases"]
        pci = set()
        for device,(pid,start) in enumerate(leader["owners"]):
            rank = node*8+device
            row = json.loads((OUT/f"owner-{rank:02}.json").read_bytes())
            require(row["rank"]==rank and row["device"]==device and row["step"]==identity["step"]
                    and row["owner_PID"]==pid and row["startticks"]==start and
                    row["passed"] and row["normal_close"] and row["both_ccl_initialized"],
                    "rank did not complete")
            ids = row["UID_SHA256"]
            require(len(ids)==2 and ids[0]!=ids[1] and all(re.fullmatch("[0-9a-f]{64}", x) for x in ids)
                    and uids in (None, ids), "CCL identity disagrees across owners")
            uids = ids
            pci.add(row["physical_PCI"])
            if device==0:
                expected=[(m,p,False,"numbers") for m,p in ((65,64),(1313,1312),(2048,32767))]
                expected.extend((m,p,order,"timing") for m,p in ((65,64),(1313,1312),(2048,32767))
                                for order in (False,True,True,False)*4)
                records=row["records"]
                require([(r["rows"],r["last_visible"],r["packed_first"],r["phase"]) for r in records]
                        ==expected and len(row["plan"])==51 and all(
                        r["rc"]==0 and r["elements"]==r["rows"]*1024 and
                        not any(r[k] for k in ("different_bits","nonfinite_original","nonfinite_packed"))
                        and r["original_ns"]>0 and r["packed_ns"]>0 and r["packed_cached_dispatch"] in (0,1)
                        for r in records), "rank numerical or timing evidence differs from scope")
                report[str(rank)] = dict(records=records, input_SHA256=[
                    row["latent_SHA256"],row["rotary_SHA256"],*row["index_SHA256"]])
                require(report[str(rank)]["input_SHA256"]==[
                    "5d20804465c886cf01166179eb9eaa8129c85d74c6359c01dc3b75772fef1269",
                    "fb5535e65c098b44ea0e91eee66a6ca572604932d07e6feb9fb07735f7cec56e",
                    "7bb4b16bbdc1c6ef40694d9ec165b4c114ae0bf0d9b863130560fa99163585f9",
                    "b1911c808a93c7e650bc963ff687aed6363ad0dd8088b06c50c5a6fd975cc92d",
                    "3e96c1a31b0f26dead5bf07bd0f73f4046e633c71096c3751a7e68f6b9933a19"],
                    "device caller did not use the frozen controlled fixture")
        require(set(gate[name]["pci_to_kfd"])==pci and len(pci)==8,"owner PCI/KFD mapping differs")
        cards += len(pci)
    require(report["0"]["input_SHA256"]==report["8"]["input_SHA256"],"measured ranks saw different inputs")
    return dict(owners=cards, phases=phases, measured=report)


def execute(claim, request, epoch):
    remaining = epoch + 360 - time.time()
    require(330<remaining<=360, "unexpected initial deadline")
    end = time.monotonic()+remaining
    final = dict(claim=claim,request_id=request,epoch=epoch,passed=False,
                 scope="TP16 selected-value component, controlled BF16 caches", stages={})
    clean = {k:v for k,v in os.environ.items() if not k.startswith(
             ("SLURM_","PMI_","PMIX_","CUDA_","HIP_","ROCR_","GPU_")) or k in {"SLURM_CONF","SLURM_CONF_SERVER"}}
    clean.update(GLM_SELECTED_CLAIM=claim,GLM_SELECTED_EPOCH=str(epoch))
    try:
        preflight(end-300)
        for phase in ("pre","gate","post"):
            (OUT/phase).mkdir(mode=0o700)
        pre = step("pre",end-270,end-292,clean)
        final["stages"]["pre"] = pre
        before = snapshots("pre",pre)
        gpu = step("gate",end-68,end-95,clean)
        final["stages"]["gate"] = gpu
        require(gpu["state"]=="TERMINAL","GPU accounting unresolved")
        post = step("post",end-1,end-27,clean)
        final["stages"]["post"] = post
        after = snapshots("post",post)
        for node in NODES:
            require(after[node]["boot_id"]==before[node]["boot_id"] and after[node]["pci"]==before[node]["pci"]
                    and after[node]["monotonic_ns"]>=before[node]["monotonic_ns"],"post-run identity changed")
        final["post_release_verified"] = True
        final["component"] = audit_owners(gpu,snapshots("gate",gpu),claim,epoch)
        final["order_adjusted"] = summarize_order(final["component"]["measured"])
        final["passed"] = True
    except Exception as exc:
        final["error"] = str(exc)[:240]
    finally:
        save(OUT/"outcome.json",final)
    return final
