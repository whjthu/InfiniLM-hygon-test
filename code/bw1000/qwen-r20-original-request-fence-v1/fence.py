"""One original Qwen payload fence and its independent read-only reconciler."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time


REQUEST="63fcb8fcab2e4d77897ddcab6ee72f90"
PARENT,NODE,UID="1103606","f13r1n09",154459
RUN=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")/("qwen-textvqa-first-two-"+REQUEST)
GATE="qwen-r20-v5-live"
MARKER="cancelled-original-request.json"
ORIGINAL_PROGRAM="d75da53ceb4b14b561ded99795f7f5598cae2e189e9f6e580ddae345881fd0d9"
ORIGINAL_PAYLOAD="23a0a09ea464aca9d34fc628a8d13ae12f790babcf1ed187270ee5a0c8630c6d"
SOURCE_SHA="dcc6e63f692f552caf28b14d008d0547b55f47cb73066bb5d345389f4499c6d2"
PACKET_SHA="a29f8cb39573678c98f1d0bebcb77ebec36e92d218366cc605a6dd89b9ba3939"
ASSETS_SHA="231aa69743a51e52986838899b1e57cdd83dabe343ad9cb8bbb56dacda06d208"
IMPORT_SHA="0a2618abffdf31903ba6f38748ae6af65e4d3702873cfd866ad46594404e25ba"
IMPORT_REQUEST="63cd98bbf5814ca1a493bb6f517a76d4"
ORIGINAL_SCOPE="c084a6ec29803547be36a246e0adc052cddeadf0"
ARCHIVE_PUBLICATION="1d68110e7e54e37268e55ce605eecb594ae087b6"
ORIGINAL_LAUNCH_SHA="787f6431f509ada07f5b23d6b62d54f70b178366f61cc79a863ef750e08e59fd"
FROZEN_RECONCILE_SHA="3b2753b33a5900275bb85445b07e079e3a57dcd03c28b775357b84737edb2e6f"
NONCE="909ca6521c9242c8a4dbe315227afb5"
PROBE_LABEL="training80-admit-"+sha256(f"{NONCE}:{PARENT}:0".encode()).hexdigest()[:32]
COLUMNS="JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList%60,AllocTRES%150,Start,End"
SINCE="2026-11-25T04:34:40"
TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL","DEADLINE","REVOKED"}
OWNER_NAMES={"intent.json","owner-terminal.json","owner-outcome.json","assigned.json","runtime-provider.json",
             "never-populated-python-cache","before.json","after.json","capture","train","child-owner.json","native.log"}
OWNER_PREFIXES=("keeper-","keeper.","pre-","pre.","gpu-","gpu.","post-","post.","native-")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value,sort_keys=True,allow_nan=False,separators=(",",":"))+"\n").encode()


MARKER_BODY={
    "schema":"qwen_original_r20_pre_scope_fence.v1","request":REQUEST,"parent":PARENT,"node":NODE,"gate":GATE,
    "source_manifest_SHA256":SOURCE_SHA,"original_scope_publication":ORIGINAL_SCOPE,"archive_publication":ARCHIVE_PUBLICATION,
    "original_launch_SHA256":ORIGINAL_LAUNCH_SHA,"original_program_SHA256":ORIGINAL_PROGRAM,
    "original_payload_SHA256":ORIGINAL_PAYLOAD,"replay_allowed":False,"scope_rename_allowed":False,
    "GPU_launch_allowed":False,"resource_release_authorized":False}
FENCE_ID=digest(encode(MARKER_BODY))[:32]
MARKER_BYTES=encode(dict(MARKER_BODY,fence_id=FENCE_ID))
MARKER_SHA=digest(MARKER_BYTES)


def file_bytes(path,maximum,expected=None):
    info=path.lstat()
    need(stat.S_ISREG(info.st_mode) and path.resolve(strict=True)==path
         and 0<info.st_size<=maximum,"EXACT_FENCE_READ_ONLY_FILE")
    with path.open("rb") as stream:
        raw=stream.read(maximum+1)
    need(len(raw)==info.st_size and (expected is None or digest(raw)==expected),"EXACT_FENCE_READ_ONLY_BYTES")
    return raw


def original_stage(run):
    need(sys.platform=="linux" and os.getuid()==UID and run.resolve(strict=True)==run
         and run.is_dir() and run.stat().st_uid==UID,"EXACT_ORIGINAL_TARGET_HOST_OWNER")
    src=run/"src"
    value=json.loads(file_bytes(src/"PACKET.json",32768,SOURCE_SHA))
    need(len(value["files"])==20,"EXACT_INSTALLED_V5_SOURCE_CLOSURE")
    for name,pin in value["files"].items():
        need(len(file_bytes(src/name,pin["bytes"],pin["SHA256"]))==pin["bytes"],"EXACT_INSTALLED_V5_SOURCE_MEMBER")
    stage=json.loads(file_bytes(run/"install.json",4096))
    assets=json.loads(file_bytes(run/"assets-rehashed.json",262144,ASSETS_SHA))
    file_bytes(run/("py310-import-"+IMPORT_REQUEST+".json"),262144,IMPORT_SHA)
    need(stage["request"]==REQUEST and stage["source_manifest_SHA256"]==SOURCE_SHA and stage["source_files"]==20
         and stage["state"]=="EXACT_SOURCE_ONLY_PACKET_INSTALLED"
         and assets["model_files"]==14 and assets["framework_files"]==393,"EXACT_V5_TARGET_ASSET_AND_IMPORT_AUTHORITY")


def occupants(run):
    return sorted(path.name for path in run.iterdir()
                  if path.name in OWNER_NAMES or path.name.startswith(OWNER_PREFIXES))


def gate_state(run):
    gate=run/GATE
    try:
        info=gate.lstat()
    except FileNotFoundError:
        return "ABSENT"
    if not stat.S_ISDIR(info.st_mode) or gate.resolve(strict=True)!=gate:
        return "EXISTING_NON_FENCE_ENTRY"
    if os.name=="posix" and (info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700):
        return "EXISTING_DIRECTORY_WITH_OTHER_OWNER_OR_MODE"
    try:
        values=sorted(path.name for path in gate.iterdir())
        if values==[MARKER] and file_bytes(gate/MARKER,4096,MARKER_SHA)==MARKER_BYTES:
            return "EXACT_CANCEL_MARKER"
    except (FileNotFoundError,OSError,ValueError):
        return "OCCUPIED_OR_PARTIAL"
    return "OCCUPIED_OR_PARTIAL"


def query(*argv):
    need(argv[0] in {"scontrol","squeue","sacct"},"READ_ONLY_SCHEDULER_COMMANDS")
    output=subprocess.run(argv,check=True,capture_output=True,stdin=subprocess.DEVNULL,timeout=7,
                          env=dict(os.environ,TZ="UTC",SLURM_TIME_FORMAT="%Y-%m-%dT%H:%M:%S")).stdout
    need(len(output)<=1048576,"BOUNDED_READ_ONLY_SCHEDULER_RESULT")
    return output.decode("ascii").splitlines()


def fields(line):
    return dict(part.split("=",1) for part in line.split() if "=" in part)


def scheduler(read=query,clock=time.monotonic):
    started=clock()
    seen={}
    def hosts(expr):
        if expr not in seen:
            need(re.fullmatch(r"[A-Za-z0-9\[\],._-]+",expr or "") and expr not in {"None","Unknown","null"},
                 "EXACT_READ_ONLY_SCHEDULER_NODELIST")
            entries=[expr] if re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+",expr) else read("scontrol","show","hostnames",expr)
            need(entries and len(set(entries))==len(entries),"EXACT_READ_ONLY_SCHEDULER_NODES")
            seen[expr]=set(entries)
        need(clock()-started<26,"STALE_SCHEDULER_READ")
        return seen[expr]

    [row]=read("scontrol","show","job",PARENT,"-o")
    job=fields(row)
    members=hosts(job["NodeList"])
    need(job["JobId"]==PARENT and job["JobState"]=="RUNNING" and job["UserId"].endswith(f"({UID})")
         and job["NumNodes"]=="40" and len(members)==40 and NODE in members
         and "gres/dcu=320" in job["AllocTRES"].split(","),"EXACT_CURRENT_PARENT_NODE_MEMBERSHIP")

    def original(label):
        return REQUEST in label
    def related(name,where):
        return original(name) or name==PROBE_LABEL or NODE in hosts(where)
    def numeric(step):
        return re.fullmatch(PARENT+r"\.[0-9]+",step) is not None

    control,queue,original_rows,probe_rows,other_active=[],[],[],[],[]
    current=read("scontrol","-o","show","step",PARENT)
    need(current,"AUTHORITATIVE_SCONTROL_STEP_READ_REQUIRED")
    for data in map(fields,current):
        step=data["StepId"]
        if step in {PARENT+".batch",PARENT+".extern"}:
            continue
        need(numeric(step),"EXACT_ACTIVE_SCONTROL_NUMERIC_STEP")
        if related(data["Name"],data["NodeList"]):
            control.append([step,data["Name"],data["NodeList"],data["State"]])
    for row in read("squeue","--steps","--jobs="+PARENT,"-h","-o","%i|%j|%N"):
        step,name,where=row.split("|")
        if step in {PARENT+".batch",PARENT+".extern"}:
            continue
        need(numeric(step),"EXACT_ACTIVE_SQUEUE_NUMERIC_STEP")
        if related(name,where):
            queue.append([step,name,where])
    end=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    for line in read("sacct","-nP","-j",PARENT,"-S",SINCE,"-E",end,"--format",COLUMNS):
        row=line.split("|")
        need(len(row)==9,"EXACT_NINE_ACCOUNTING_COLUMNS")
        if not numeric(row[0]):
            continue
        state=row[2].split()[0]
        if original(row[1]):
            original_rows.append(row)
        elif row[1]==PROBE_LABEL:
            probe_rows.append(row)
        elif state not in TERMINAL and NODE in hosts(row[5]):
            other_active.append(row)
    need(clock()-started<26,"STALE_SCHEDULER_READ")
    return dict(scontrol=control,squeue=queue,original_accounting=original_rows,collector_accounting=probe_rows,
                other_active_on_node=other_active)


def blocked(seen):
    return any(seen[key] for key in ("scontrol","squeue","original_accounting","other_active_on_node"))


def closed_probe(seen):
    rows=seen["collector_accounting"]
    return len(rows)<=1 and all(row[2].split()[0] in TERMINAL and row[5]==NODE
        and row[6]=="cpu=1,mem=256M,node=1" for row in rows)


def sync_directory(path):
    if os.name=="posix":
        fd=os.open(path,os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_marker(gate):
    with (gate/MARKER).open("xb") as stream:
        stream.write(MARKER_BYTES)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(gate)


def create_fence(run,observe=scheduler,write=write_marker,clock=time.monotonic):
    started=clock()
    if gate_state(run)!="ABSENT" or occupants(run):
        return {"state":"ORIGINAL_AWAKE_OR_GATE_OCCUPIED_NO_WRITE","marker_created":False}
    seen=observe()
    if blocked(seen) or not closed_probe(seen) or occupants(run) or clock()-started>=28:
        return {"state":"FRESH_ORIGINAL_SCHEDULER_OR_FILE_BOUNDARY_NOT_CLEAR","marker_created":False,
                "scheduler":seen}
    gate=run/GATE
    try:
        gate.mkdir(mode=0o700)
    except FileExistsError:
        return {"state":"ATOMIC_CREATE_LOST_ORIGINAL_OR_FENCE_OWNS_DIRECTORY","marker_created":False}
    try:
        sync_directory(run)
        write(gate)
        need(gate_state(run)=="EXACT_CANCEL_MARKER","EXACT_MARKER_IMMEDIATE_READBACK")
    except (OSError,ValueError) as err:
        return {"state":"ATOMIC_DIRECTORY_CREATED_BUT_MARKER_UNQUALIFIED","marker_created":False,
                "directory_created":True,"error_type":type(err).__name__}
    return {"state":"ORIGINAL_PRE_SCOPE_GATE_WON_CANCEL_MARKER_DURABLE","marker_created":True,
            "marker_SHA256":MARKER_SHA,"fresh_observation_SHA256":digest(encode(seen))}


def independently_read(run,observe=scheduler):
    gate=gate_state(run)
    before=occupants(run)
    seen=observe()
    after=occupants(run)
    stable=gate==gate_state(run) and before==after
    qualified=stable and gate=="EXACT_CANCEL_MARKER" and not after and not blocked(seen) and closed_probe(seen)
    return {"state":"EXACT_ATOMIC_ORIGINAL_REQUEST_NO_REPLAY_FENCE_READBACK" if qualified else "ORIGINAL_KEEP_UNKNOWN_OR_BUSY",
            "gate":gate,"original_owner_files":after,"scheduler":seen,"stable_read":stable,
            "original_payload_blocked_before_detach":qualified,"marker_SHA256":MARKER_SHA if qualified else None,
            "writes":0}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode",choices=("apply","read-back"))
    cli.add_argument("--fence-id",required=True)
    args=cli.parse_args()
    need(args.fence_id==FENCE_ID,"EXACT_ORIGINAL_FENCE_ID")
    original_stage(RUN)
    os.umask(0o077)
    result=create_fence(RUN) if args.mode=="apply" else independently_read(RUN)
    result.update(request=REQUEST,fence_id=FENCE_ID,replay_allowed=False,scope_rename_allowed=False,
                  GPU_authorized=False,release_authorized=False,new_scheduling_operations=0)
    print(encode(result).decode().strip())


if __name__=="__main__":
    main()
