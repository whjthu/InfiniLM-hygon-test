"""Single authorized CPU index launch using the exact R27 co-resident guard."""

import argparse
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import time

from cpu_command import cpu_argv
from prepare import LABEL,NEW,REQUEST,digest,emit_file,encoded,need,pin,read


SOURCE=Path(__file__).resolve().parent
STANDBY_STEP="1103606.3355"
STANDBY_LABEL="training80-dcu-standby-4d63273bbce5-f13r1n12"
SLOTS_SHA="9c11255b163faf83aa9c2fc2b528312f2939900a81b07f122a0f234dd2ea7db4"


def source_check(expected):
    need(SOURCE==NEW/"r39-source","R39_EXACT_CPU_SOURCE_TARGET")
    manifest=json.loads(read(SOURCE/"CPU_SOURCE_FILES.json",expected,16384))
    for name,record in manifest["files"].items():
        need(Path(name).name==name,"R39_SAFE_CPU_SOURCE_BASENAME")
        read(SOURCE/name,record["SHA256"],65536,record["bytes"])
    return manifest


def host(args):
    source_check(args.source_sha256)
    need(os.getuid()==154459 and re.fullmatch(r"[0-9a-f]{40}",args.publication)
         and not os.path.lexists(NEW/"INDEX_INTENT.json") and not os.path.lexists(NEW/"INDEX_RESULT.json"),
         "R39_EXACT_CPU_OWNER_NO_REPLAY")
    raw=read(NEW/"STAGE_VIEW.json",args.stage_view_sha256,16384)
    planned=json.loads(raw)
    need(planned["request"]==REQUEST and planned["old_bytes_unchanged"]==788 and planned["target_GPU"]==0
         and planned["actual_optimizer_updates"]==0,"R39_ACTUAL_STAGED_12_OF_800")
    read(NEW/"lineage-private.json",planned["lineage"]["SHA256"],65536,planned["lineage"]["bytes"])
    read(SOURCE/"cpu_slots.py",SLOTS_SHA,65536)
    original=runpy.run_path(str(SOURCE/"cpu_slots.py"))
    fresh=original["fresh_slots"]
    name=LABEL+"-index-"+REQUEST[:12]
    fresh.__globals__.update(STANDBY_STEP=STANDBY_STEP,STANDBY_LABEL=STANDBY_LABEL,JOB=name,REQUEST=REQUEST,RUN=NEW)
    env=original["host_environment"]()
    query_deadline=time.monotonic()+30
    def query(*cmd):
        need(cmd[0] in ("sacct","scontrol","squeue"),"R39_READ_ONLY_SCHEDULER_PRECHECK")
        left=query_deadline-time.monotonic()
        need(left>0,"R39_SCHEDULER_PRECHECK_BOUND")
        result=subprocess.run(cmd,stdin=subprocess.DEVNULL,capture_output=True,check=True,timeout=min(8,left),env=env)
        need(len(result.stdout)<=1048576,"R39_BOUNDED_SCHEDULER_METADATA")
        return result.stdout.decode("ascii").splitlines()
    def absent():
        members=query("sacct","-nP","-j","1103606","--format","JobIDRaw,JobName%120")
        need(all(len(line.split("|"))==2 for line in members) and not any(line.split("|")[1]==name for line in members),
             "R39_ORIGINAL_CPU_LABEL_UNIQUENESS")
    absent()
    slots=fresh(query,time.time())
    absent()
    epoch=time.time()
    need(epoch+240<=slots["parent_end_epoch"],"R39_FULL_240_SECOND_CPU_PARENT_WINDOW")
    argv=cpu_argv(epoch,planned["lineage"]["SHA256"])
    need(argv[0]=="srun" and "--gres=none" in argv and "--overlap" not in argv and "--exact" in argv,"R39_ZERO_GRES_SINGLE_INDEX_ARGV")
    os.umask(0o077)
    intent={"request":REQUEST,"publication":args.publication,"epoch":epoch,"deadline":epoch+240,
            "source_manifest_SHA256":args.source_sha256,"stage_view_SHA256":args.stage_view_sha256,
            "argv_SHA256":digest(encoded(argv)),"replay_allowed":False,"GPU":0,"placement":slots}
    emit_file(NEW/"INDEX_INTENT.json",encoded(intent))
    record={"request":REQUEST,"intent":pin(encoded(intent)),"attempted":False,"GPU":0,"replay_allowed":False}
    try:
        with (NEW/"index-host.stdout").open("xb") as out,(NEW/"index-host.stderr").open("xb") as err:
            left=epoch+240-time.time()
            need(left>0,"R39_CPU_INDEX_EPOCH_EXHAUSTED")
            record["attempted"]=True
            child=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=left,env=env,check=False)
            record.update(rc=child.returncode,state="SRUN_RETURNED_REQUIRES_INDEPENDENT_TERMINAL")
    except (OSError,subprocess.SubprocessError,ValueError) as error:
        record.update(state="RECONCILE_ORIGINAL_UNKNOWN",error_class=type(error).__name__)
    emit_file(NEW/"INDEX_HOST_RETURN.json",encoded(record))
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("publication","source-sha256","stage-view-sha256"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    print(encoded(host(a)).decode(),end="")
