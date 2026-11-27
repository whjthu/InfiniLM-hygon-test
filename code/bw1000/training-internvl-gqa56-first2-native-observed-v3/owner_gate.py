"""Validate an explicit fresh owner and frozen CPU qualification for one GQA56 trial."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess


REQUEST="7d1385cd4a944f74bfa54079ce9cce10"
LABEL="training-internvl-gqa56-first2-native-observed-v3"
PLACEMENT=dict(parent="1103606",node="f13r1n14",dcu=8,cpus=64,mem_mib=224000,port=29987)
HOLDER="1103606.4636"
EXPECTED_SHA="1241b04ceb4a978f2e433ead279c37bc570c0f57670b11fc7dc1b7750952d3a9"
WORKER_SHA="eb389f50a9b5a33c953f7d94e3ee20394c8aa2d00724a31fb4fdda84a53765b3"
CPU_STAGED=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"
                "training-internvl-gqa56-first2-native-observed-v2-source-883dca77519d4ef2af89a941248fca12/cpu-result.json")
CPU_SHA="756188947ce2e554feb39ff691e908b7483c76a3c8263c48ad649e2e366e4757"
CPU_ARCHIVE_SHA="f11ac48afbf62dca65c3c08e375a3b72e9dc4eb84d73e068f32c0dc3b2fe693a"
CPU_ROW=["1103606.4701","internvl-gqa56-first2-preflight-e13fe8785ca9","COMPLETED","0:0","1","f13r1n14",
         "cpu=2,mem=4G,node=1","2026-11-27T17:17:37","2026-11-27T17:17:38"]
COLS="JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
HEX64=re.compile(r"[a-f0-9]{64}")


def need(ok,code):
    if not ok:
        raise ValueError(code)


def clock(value):
    return type(value) in (int,float) and math.isfinite(value)


def record_epoch(value):
    need(isinstance(value,str) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d",value),"EXACT_UTC_ACCOUNTING_TIME")
    return datetime.fromisoformat(value+"+00:00").timestamp()


def validate_release(value,issued):
    need(set(value)=={"request","parent","node","holder_step","accounting","expected_file_SHA256",
                      "worker_file_SHA256","worker_returned","numbered_steps","queued_steps","observed_epoch"} and
         (value["request"],value["parent"],value["node"],value["holder_step"])==
         (REQUEST,PLACEMENT["parent"],PLACEMENT["node"],HOLDER) and value["worker_returned"] is True and
         value["numbered_steps"]==value["queued_steps"]==[] and value["expected_file_SHA256"]==EXPECTED_SHA and
         value["worker_file_SHA256"]==WORKER_SHA,"EXACT_PRIOR_STANDBY_RELEASE_EVIDENCE")
    row=value["accounting"]
    need(type(row) is list and len(row)==9 and
         row[:4]==[HOLDER,"training80-dcu-standby-9fd58db691fe-f13r1n14","COMPLETED","0:0"] and
         re.fullmatch(r"[0-9]+",row[4]) and row[5:7]==[PLACEMENT["node"],"cpu=8,gres/dcu=8,mem=16G,node=1"],
         "EXACT_OWNED_STANDBY_TERMINAL_ROW")
    need(clock(value["observed_epoch"]) and record_epoch(row[7])<=record_epoch(row[8])<=value["observed_epoch"]<=issued,
         "EXACT_STANDBY_RELEASE_CHRONOLOGY")


def validate_owner(value,publication,claim_sha,manifest_sha,now):
    need(set(value)=={"request","label","publication","claim_SHA256","manifest_SHA256","placement","zero_gres_step",
                      "zero_gres_result_SHA256","release_evidence_SHA256","issued_epoch","expires_epoch",
                      "GPU_GO","replay_allowed"} and
         (value["request"],value["label"],value["publication"],value["claim_SHA256"],value["manifest_SHA256"])==
         (REQUEST,LABEL,publication,claim_sha,manifest_sha) and
         re.fullmatch(r"[a-f0-9]{40}",publication) and all(HEX64.fullmatch(x) for x in
         (claim_sha,manifest_sha,value["release_evidence_SHA256"])) and
         value["placement"]==PLACEMENT and value["zero_gres_step"]==CPU_ROW[0] and
         value["zero_gres_result_SHA256"]==CPU_SHA and value["GPU_GO"] is True and value["replay_allowed"] is False,
         "EXACT_EXPLICIT_OWNER_AUTHORITY")
    issued,expires=value["issued_epoch"],value["expires_epoch"]
    need(clock(issued) and clock(expires) and issued<=now<expires==issued+120,"EXACT_FRESH_OWNER_WINDOW")


def load(here,authority_sha,publication,claim_sha,manifest_sha,pin,now):
    need(isinstance(authority_sha,str) and HEX64.fullmatch(authority_sha),"EXACT_OWNER_PIN_REQUIRED")
    value=json.loads(pin(here/"owner-authority.json",expected=authority_sha,cap=8192))
    validate_owner(value,publication,claim_sha,manifest_sha,now)
    release=json.loads(pin(here/"holder-release.json",expected=value["release_evidence_SHA256"],cap=8192))
    validate_release(release,value["issued_epoch"])
    pin(CPU_STAGED,expected=CPU_SHA,cap=32768)
    return value


def verify_cpu_accounting(run=subprocess.run):
    argv=["sacct","-nP","-j",CPU_ROW[0],"-o",COLS]
    env={**os.environ,"TZ":"UTC","LC_ALL":"C","SLURM_TIME_FORMAT":"standard"}
    result=run(argv,stdin=subprocess.DEVNULL,capture_output=True,timeout=8,check=False,env=env)
    need(result.returncode==0 and not result.stderr and len(result.stdout)<=65536,"EXACT_QUIET_ZERO_GRES_ACCOUNTING")
    rows=[line.split("|") for line in result.stdout.decode().splitlines() if line.split("|")[0]==CPU_ROW[0]]
    need(rows==[CPU_ROW],"EXACT_ORIGINAL_4701_ZERO_GRES_TERMINAL")
    return rows[0]
