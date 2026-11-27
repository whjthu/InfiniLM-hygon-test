"""Verify an unprofiled T09 timing repetition from independent stored native bytes."""
import json
import math
import os
import re
import subprocess

from placement import TERMINAL, step_tres, utc


BOUNDARY=("before first measured update through completed final checkpoint including checkpoint hashes; "
          "excludes loading, warmup, final checkpoint validation, recovery and adapter export")
COLUMNS="JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"


def need(ok,code):
    if not ok:
        raise ValueError(code)


def measured_native(output, read):
    raw,steps_pin=read(output/"steps.ndjson",65536)
    payload,manifest_pin=read(output/"manifest.json",256<<10)
    rows=[json.loads(line) for line in raw.splitlines()]
    manifest=json.loads(payload)
    metrics=manifest["metrics"]
    positive=lambda v:type(v) in (float,int) and math.isfinite(v) and v>0
    need([row["global_step"] for row in rows]==list(range(1,26)) and metrics["configured_warmup_steps"]==5
         and metrics["configured_measured_steps"]==20 and metrics["timing_boundary"]==BOUNDARY
         and manifest["runtime"]["accelerator"]=="hip","T09_NATIVE_SAME_25_5_20_BOUNDARY")
    need(all(positive(row[k]) for row in rows for k in ("started_unix","ended_unix","step_seconds",
             "forward_seconds","backward_update_seconds")),"T09_FINITE_RECORDED_UPDATE_WINDOWS")
    need(all(math.isclose(r["ended_unix"]-r["started_unix"],r["step_seconds"],rel_tol=.005)
             and r["forward_seconds"]+r["backward_update_seconds"]<=r["step_seconds"] for r in rows),
         "T09_RECORDED_PHASE_ACTUAL_BOUNDARIES")
    measured=rows[5:]
    gaps=[right["started_unix"]-left["ended_unix"] for left,right in zip(measured,measured[1:])]
    need(min(gaps)>=0,"T09_DISJOINT_NATIVE_UPDATE_TIMESTAMPS")
    rate=metrics["measurement_updates_per_second"]
    wall=metrics["measurement_wall_seconds"]
    need(positive(rate) and positive(wall) and math.isclose(wall,20/rate,rel_tol=1e-6),
         "T09_ORIGINAL_CHECKPOINT_INCLUSIVE_COUNTER")
    native_span=measured[-1]["ended_unix"]-measured[0]["started_unix"]
    need(math.fsum(r["step_seconds"] for r in measured)<=native_span+.001<=wall+.0015,
         "T09_FINAL_CHECKPOINT_END_AFTER_MEASURED_UPDATE")
    stage=manifest["stage_timings"]
    for name in ("forward_seconds","backward_update_seconds"):
        need([r[name] for r in measured]==stage[name],"T09_ALIGNED_NATIVE_STEP_STAGE_TIMINGS")
    return dict(repetition=3,first_measured_step=6,last_measured_step=25,updates=20,warmup=5,
                checkpoint_steps=[10,20,25],timing_boundary=BOUNDARY,unprofiled=True,wall_seconds=wall,
                updates_per_second=rate,recorded_update_span=native_span,final_checkpoint_remainder=wall-native_span,
                steps_pin=steps_pin,manifest_pin=manifest_pin,formal_performance_qualified=False)


def independent_rows(parent, query=subprocess.check_output):
    result=query(["env","TZ=UTC","SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S","sacct","-nP",
                  "-j",parent,"--format",COLUMNS],text=True,timeout=10)
    need(0<len(result)<=2<<20,"T09_INDEPENDENT_ACCOUNTING_EXTENT")
    rows=[line.split("|") for line in result.splitlines() if line]
    need(len(rows)<=8192 and all(len(row)==9 for row in rows),"T09_NINE_INDEPENDENT_ACCOUNTING_COLUMNS")
    return rows


def check_independent(rows, receipts, parent,node,label,epoch,deadline):
    result={}
    latest=epoch-1
    maximum={"pre":60,"gpu":1200,"post":60}
    for phase in ("pre","gpu","post"):
        stored=receipts[phase]["row"]
        name=label+"-"+phase
        candidates=[row for row in rows if row[1]==name]
        row,=candidates
        need(row[:7]==stored and re.fullmatch(parent+r"\.[0-9]+",row[0]) is not None and row[5]==node
             and step_tres(row[6],phase) and row[2] in TERMINAL
             and (phase=="gpu" or row[2:4]==["COMPLETED","0:0"]),"T09_INDEPENDENT_REAL_PHASE_AND_TRES")
        start,end=map(utc,row[7:])
        need(row[4].isdecimal() and 0<=int(row[4])<=maximum[phase] and latest<=start<=end<=deadline+1
             and abs(end-start-int(row[4]))<=1,"T09_INDEPENDENT_PHASE_ORIGINAL_BUDGET")
        latest=end
        result[phase]=row
    return result
