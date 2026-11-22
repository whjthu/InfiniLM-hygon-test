"""Submit the exact fresh keeper once, then reconcile original Slurm identities."""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from contract import (GPU, HERE, INSIDE, KEEP, LABEL, NODES, PARENT, PHASES, WHOLE, budget, need,
                      pinned, proc_start, put, read_json, sha, step_identity, validate_claim, verify_stage)
from node import hardware_matches


PYTHON = "/usr/local/python3.12/bin/python3"
COLUMNS = ["JobIDRaw","JobName%100","State","ExitCode","ElapsedRaw","NodeList","AllocTRES"]
ROLES = ("keeper", "pre", "device", "post")


def query(argv, seconds=10):
    raw = subprocess.check_output(argv,timeout=seconds,text=True)
    need(len(raw) < 2**21, "Slurm output exceeded bounded query")
    return raw.splitlines()


def expand(token):
    need(re.fullmatch(r"[A-Za-z0-9_,\[\]-]+",token), "invalid Slurm node list")
    rows = query(["scontrol","show","hostnames",token])
    need(rows and len(rows) == len(set(rows)) and all(re.fullmatch(r"f\d+r\d+n\d+", name) for name in rows),
         "unresolved Slurm hosts")
    return set(rows)


def placement(claim, allowed_step=None):
    q = query(["squeue","-h","-j",PARENT,"-o","%T|%N"])
    need(len(q) == 1,"original long allocation missing")
    state,nodes = q[0].split("|")
    selected = set(NODES) | {claim["keeper_node"]}
    need(state == "RUNNING" and selected <= expand(nodes),"current parent no longer covers requested nodes")
    steps = query(["scontrol","show","step",PARENT,"-o"])
    for row in steps:
        if row == "No steps in the system":
            need(len(steps) == 1, "contradictory steps output")
            continue
        fields = dict(word.split("=",1) for word in row.split() if "=" in word)
        sid = fields["StepId"]
        need(re.fullmatch(PARENT+r"\.(?:\d+|batch|extern)",sid),"foreign or malformed active step")
        if sid == allowed_step or sid.endswith((".batch",".extern")):
            continue
        need(not selected.intersection(expand(fields["NodeList"])), "other live Slurm step overlaps new scope")


def account(role, seconds=10, read=query):
    label = LABEL + "-" + role
    lines = read(["sacct","-n","-P","-j",PARENT,"--format="+",".join(COLUMNS)],seconds)
    rows = [line.split("|") for line in lines]
    matches = [row for row in rows if len(row) > 1 and row[1] == label]
    need(len(matches) <= 1,"duplicate exact scope step label")
    if matches:
        row = matches[0]
        need(len(row) == len(COLUMNS) and re.fullmatch(PARENT+r"\.\d+",row[0]),"invalid original accounting row")
        return row
    return None


def matching_resources(row, role, claim, hosts=expand):
    nodes = [claim["keeper_node"]] if role == "keeper" else NODES
    cpu = {"keeper":2, "pre":2,"device":128,"post":2}[role]
    memory = {"keeper":8192,"pre":2048,"device":448000,"post":2048}[role]
    tres = dict(item.split("=",1) for item in row[6].split(","))
    picked = 16 if role == "device" else 0
    gres = [int(v) for k,v in tres.items() if k.startswith("gres/") and k.count(":") == 0]
    size = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT])",tres["mem"])
    need(size, "unrecognized actual memory unit in accounting")
    mib = Decimal(size[1]) * {"K":Decimal(1)/1024,"M":1,"G":1024,"T":1024**2}[size[2]]
    need(hosts(row[5]) == set(nodes) and int(tres["cpu"]) == cpu and int(tres["node"]) == len(nodes)
         and mib == memory and sum(gres) == picked
         and (role != "device" or tres.get("gres/dcu") == "16"), "Slurm placement / allocated CPU / DCU differs")


def command(role, claim, out, epoch):
    keeper = role == "keeper"
    gpu = role == "device"
    nodes = [claim["keeper_node"]] if keeper else NODES
    times = {"keeper":KEEP,"device":GPU,"pre":60,"post":60}
    cap = WHOLE if role == "post" else KEEP if keeper else GPU
    wall = min(times[role],int(budget(epoch,cap)))
    need(wall >= 30,"insufficient original Slurm step budget")
    minutes = wall//60 if wall>=60 else 0
    slurm_wall = f"{minutes}:{wall%60:02d}" if minutes else f"0:{wall:02d}"
    action = ["host.py","keep",str(out)] if keeper else ["node.py",role,str(out)]
    return ["srun","--jobid="+PARENT,"--job-name="+LABEL+"-"+role,"--exact","--immediate=8",
            "--nodes="+str(len(nodes)),"--ntasks="+str(len(nodes)),"--ntasks-per-node=1","--distribution=block:block",
            "--kill-on-bad-exit=1","--nodelist="+",".join(nodes),"--cpu-bind="+("none" if gpu else "threads"),
            "--cpus-per-task="+ ("2" if keeper else "64" if gpu else "1"),
            "--mem="+("8192M" if keeper else "224000M" if gpu else "1024M"),
            "--gres="+("dcu:8" if gpu else "none"),"--time="+slurm_wall,
            PYTHON,"-B","-u",str(HERE/action[0]),*action[1:]]


def clean_env():
    return {key:value for key,value in os.environ.items()
            if not key.startswith(("SLURM_","PMI_","PMIX_","CUDA_","HIP_","ROCR_","GPU_","LD_PRELOAD","LD_AUDIT"))
            or key in {"SLURM_CONF","SLURM_CONF_SERVER"}}


def safe_stop(child,ticks):
    if child.poll() is not None:
        return
    need(proc_start(child.pid) == ticks and os.getpgid(child.pid) == child.pid, "owned srun PID identity changed")
    os.killpg(child.pid,signal.SIGTERM)
    try:
        child.wait(timeout=6)
    except subprocess.TimeoutExpired:
        if child.poll() is None:
            need(proc_start(child.pid) == ticks and os.getpgid(child.pid) == child.pid, "owned srun group changed")
            os.killpg(child.pid,signal.SIGKILL)
        child.wait(timeout=2)


def record_terminal(out, role, intent, result, deadline, read=account):
    from slurm_accounting import wait_terminal
    receipt = out/f"{role}-receipt.json"

    def observe():
        row = read(role,min(8,max(.01,deadline-time.monotonic())))
        result["observations"].append(row)
        put(receipt,result,replace=True)
        return row

    found = wait_terminal(observe,deadline,interval=.6,max_observations=12)
    result["row"] = found["row"]
    if found["terminal_visible"]:
        result["state"] = "TERMINAL"
        matching_resources(result["row"],role,intent["claim"])
        result["resources_bound"] = True
    return result


def perform(out, role, intent, external_cap):
    epoch = intent["epoch"]
    receipt = out/f"{role}-receipt.json"
    result = dict(role=role,state="SUBMISSION_POSSIBLE",cols=COLUMNS,observations=[],row=None)
    need(account(role) is None,"one origin step already observed")
    argv = command(role,intent["claim"],out,epoch)
    put(receipt,result)
    child = ticks = None
    try:
        with (out/f"{role}.log").open("xb") as log:
            child = subprocess.Popen(argv,env=clean_env(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ticks = proc_start(child.pid)
            result.update(srun_PID=child.pid,srun_startticks=ticks)
            put(receipt,result,replace=True)
            result["returncode"] = child.wait(timeout=min(budget(epoch,external_cap)-24,55 if role != "device" else GPU))
    except Exception as exc:
        result["error"] = str(exc)[:500]
    finally:
        if child and child.poll() is None and ticks is not None:
            try:
                safe_stop(child,ticks)
                result["forced"] = True
            except Exception as exc:
                result["cleanup_error"] = str(exc)[:500]
        result["state"] = "UNKNOWN"
        try:
            limit = time.monotonic()+min(18,budget(epoch,external_cap))
            record_terminal(out,role,intent,result,limit)
        except Exception as exc:
            result["account_error"] = str(exc)[:500]
        put(receipt,result,replace=True)
    return result


def success(receipt):
    return bool(receipt and receipt["state"] == "TERMINAL" and receipt.get("resources_bound")
                and receipt["row"][2:4] == ["COMPLETED","0:0"] and receipt.get("returncode") == 0
                and not ({"forced","error","cleanup_error","account_error"} & receipt.keys()))


def evidence(out, prefix, receipt):
    result = {}
    need(receipt["state"]=="TERMINAL" and receipt.get("resources_bound"),"exact physical phase accounting incomplete")
    for node in NODES:
        row = read_json(out/f"{prefix}-{node}.json")
        need(row["node"] == node and row["phase"] == prefix and row["actual_step"] == receipt["row"][0],
             "physical result step differs")
        result[node] = row
    return result


def progress_denominator(out, step):
    entries = []
    for rank in range(16):
        try:
            raw = read_json(out/f"owner-{rank:02d}.json")
        except FileNotFoundError:
            entries.append(dict(rank=rank,state="MISSING"))
            continue
        expected = raw["identity"]["rank"] == rank and raw["run"]["step"] == step
        report = raw["pair"]
        attempted,returned = report.get("attempted_forwards",0),report.get("successful_forwards",0)
        if expected and type(attempted) is int and type(returned) is int and 0 <= returned <= attempted <= 6:
            entries.append(dict(rank=rank,state=report["state"],attempted=attempted,returned=returned))
        else:
            entries.append(dict(rank=rank,state="REJECTED_IDENTITY_OR_COUNTS"))
    return entries


def consume_pair(out, gpu, before):
    sys.path.insert(0,str(HERE/"pair"))
    from aggregate import summarize

    admitted, owners, common = {}, {}, None
    for index,node in enumerate(NODES):
        leader = read_json(out/f"leader-{index}.json")
        need(leader["state"] == "COMPLETE" and leader["run"]["step"] == gpu["row"][0]
             and leader["identity"]["node"] == node and len(leader["spawned"]) == 8
             and [p["phase"] for p in leader["rounds"]] == list(PHASES)
             and leader["providers"]["confirmed_providers"] == 71, "leader evidence incomplete")
        need(common in (None,leader["rounds"]), "two leader ready digests differ")
        common = leader["rounds"]
        need(set(leader["logical_PCI"]) == set(before[node]["PCI"]), "permit physical mapping differs")
        for dev,pci in enumerate(leader["logical_PCI"]):
            rank = index*8+dev
            admitted[rank] = pci
            record = read_json(out/f"owner-{rank:02d}.json")
            need(record["run"] == leader["run"]
                 and all(record["identity"][key] == value for key,value in leader["spawned"][dev].items()),
                 "primary owner identity differs")
            owners[rank] = record["pair"]
    return summarize(owners,admitted)


def keep(out):
    intent = read_json(out/"intent.json")
    claim,epoch = intent["claim"],intent["epoch"]
    here = socket.gethostname().split(".")[0]
    step = step_identity()
    need(here == claim["keeper_node"] and len(os.sched_getaffinity(0)) == 2,"actual zero-GRES keeper placement")
    verify_stage(HERE,claim["source_SHA256"])
    report = dict(state="KEEPER_ENTERED",step=step,pid=os.getpid(),startticks=proc_start(os.getpid()),
                  GPU_steps=0,pre=None,device=None,post=None,resource_clear=False,
                  pair=None,prefill_pair_evidence_pass=False,owners_planned=16,forwards_planned=96)
    watch = out/"keeper-watch.json"
    put(watch,report)
    try:
        placement(claim,step)
        pre = report["pre"] = perform(out,"pre",intent,KEEP)
        need(success(pre),"fresh pre-run inventory unavailable")
        original = evidence(out,"pre",pre)
        placement(claim,step)
        gpu = report["device"] = perform(out,"device",intent,GPU+30)
        put(watch,report,replace=True)
        need(gpu["state"] == "TERMINAL" and gpu.get("resources_bound"),
             "GPU terminal or resource identity unresolved; preserve reservation")
        report["GPU_steps"] = 1
        post = report["post"] = perform(out,"post",intent,KEEP)
        need(success(post),"independent zero-GRES cleanup accounting unavailable")
        for node, later in evidence(out,"post",post).items():
            hardware_matches(original[node],later)
        report["resource_clear"] = True
        report["owner_denominator"] = progress_denominator(out,gpu["row"][0])
        need(success(gpu),"GPU failed; independent physical cleanup verified")
        leased = evidence(out,"device",gpu)
        for node,record in leased.items():
            hardware_matches(original[node],record,gate=True)
        pair = report["pair"] = consume_pair(out,gpu,leased)
        report["prefill_pair_evidence_pass"] = pair["prefill_pair_evidence_pass"]
        report["state"] = "COMPLETE" if pair["prefill_pair_evidence_pass"] else "NUMERICAL_FAILURE"
    except Exception as exc:
        report.update(state="FAILED_OR_UNRESOLVED",error=str(exc)[:512])
        raise
    finally:
        put(watch,report,replace=True)


def launch(claim_path,claim_pin,source_pin,admission_path):
    verify_stage(HERE,source_pin)
    claim = validate_claim(json.loads(pinned(claim_path,claim_pin,32768)),source_pin)
    pinned(admission_path,claim["root_admission_SHA256"],4*2**20)
    placement(claim)
    need(all(account(role) is None for role in ROLES),"original exact scope already visible in accounting")
    out = HOST/"results"/LABEL
    out.mkdir(mode=0o700)
    epoch = time.time()
    intent = dict(epoch=epoch,claim=claim,claim_SHA256=claim_pin,pid=os.getpid(),startticks=proc_start(os.getpid()))
    put(out/"intent.json",intent)
    with (out/"keeper.log").open("xb") as log:
        child = subprocess.Popen(command("keeper",claim,out,epoch),env=clean_env(),stdout=log,
                                 stderr=subprocess.STDOUT,start_new_session=True)
        handle = dict(pid=child.pid,startticks=proc_start(child.pid),intent_SHA256=sha((out/"intent.json").read_bytes()))
        put(out/"keeper-submission.json",handle)
    return dict(state="SUBMISSION_POSSIBLE",scope=str(out),**handle)


def status(out):
    from slurm_accounting import is_terminal
    need(out == HOST/"results"/LABEL, "unexpected fixed outcome namespace")
    intent = read_json(out/"intent.json")
    row = account("keeper")
    if not is_terminal(row):
        return dict(state="PENDING_OR_UNKNOWN",keeper=row,release=False)
    matching_resources(row,"keeper",intent["claim"])
    try:
        watch = read_json(out/"keeper-watch.json")
    except FileNotFoundError:
        return dict(state="KEEPER_TERMINAL_WATCH_MISSING",keeper=row,release=False)
    need(watch["step"] == row[0], "actual keeper accounting differs")
    observed = {role:account(role) for role in ("device","post")}
    release = watch["resource_clear"] and all(is_terminal(observed[role]) and
                observed[role][0] == watch[role]["row"][0] for role in observed)
    if release:
        for role in observed:
            matching_resources(observed[role],role,intent["claim"])
        release = observed["post"][2:4]==["COMPLETED","0:0"]
    passed = release and row[2:4]==["COMPLETED","0:0"] and watch["prefill_pair_evidence_pass"]
    return dict(state="TERMINAL",keeper=row,rows=observed,release=release,prefill_pair_evidence_pass=passed,
                watch_SHA256=sha((out/"keeper-watch.json").read_bytes()),GPU_GO=False,Model_GO=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode",required=True)
    start = sub.add_parser("launch")
    start.add_argument("--claim",type=Path,required=True)
    start.add_argument("--claim-sha256",required=True)
    start.add_argument("--source-sha256",required=True)
    start.add_argument("--admission",type=Path,required=True)
    sub.add_parser("status")
    keeper = sub.add_parser("keep")
    keeper.add_argument("out",type=Path)
    args = parser.parse_args()
    if args.mode == "keep":
        keep(args.out)
        return
    if args.mode == "launch":
        result = launch(args.claim,args.claim_sha256,args.source_sha256,args.admission)
    else:
        result = status(HOST/"results"/LABEL)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
