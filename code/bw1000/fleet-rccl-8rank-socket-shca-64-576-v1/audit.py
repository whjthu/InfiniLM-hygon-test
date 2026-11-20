"""Consume exact independently attested per-rank and scheduler results."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

MODES=("socket","shca","shca","socket")
NETWORK={"socket":"Socket","shca":"IBext_v8"}


def need(condition,reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    raw=path.read_bytes()
    need(0<len(raw)<4*1024**2,"RECEIPT_BOUND")
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def verify(root,pairs,request):
    terminal,pin=read(root/"dispatch-terminal.json")
    before=terminal["before"]; after=terminal["after"]
    need(terminal["request"]==request,"TERMINAL_REQUEST")
    need(after is not None and before["accounting"][2:4]==after["accounting"][2:4]==["COMPLETED","0:0"],
         "INDEPENDENT_TERMINAL_CLEANUP")
    pin_before={r["node"]:r["pci"] for r in before["nodes"]}
    pin_after={r["node"]:r["pci"] for r in after["nodes"]}
    nodes=[n for pair in pairs for n in pair]
    need(pin_before==pin_after and set(pin_before)==set(nodes)
         and all(r["holders"]==0 for r in before["nodes"]+after["nodes"]),"INDEPENDENT_PHYSICAL_CLEANUP")
    results=[]; failed=[]
    for pair in pairs:
        try:
            row=verify_pair(root,pair,terminal["arms"],pin_before)
            results.append(row)
        except (OSError,ValueError,KeyError,TypeError) as error:
            reason=str(error) if isinstance(error,ValueError) and re.fullmatch("[A-Z_0-9]+",str(error)) else type(error).__name__
            failed.append(dict(nodes=pair,reason=reason))
    return dict(planned_pairs=len(pairs),complete_pairs=len(results),planned_dcu=len(nodes)*8,
                accounting=terminal["arms"],independent_pre=before["accounting"],
                independent_post=after["accounting"],terminal_SHA256=pin,pairs=results,failures=failed,
                controller_failure=terminal["failure"])


def verify_pair(root,pair,arms,pci):
    medians=[]; receipts={}
    for i,mode in enumerate(MODES):
        step=arms[str(i)]["row"][0]
        # A globally failed step may still have independently complete per-pair receipts.
        sample={}
        for side,node in enumerate(pair):
            directory=root/f"arm-{i}"/node
            outcome,h=read(directory/"terminal.json")
            receipts[f"{i}/{node}/terminal"]=h
            need(outcome["step"]==step and outcome["mode"]==mode and outcome["cohort"]==pair
                 and not outcome["error"] and not outcome["post_kfd_holders"]
                 and outcome["exits"]=={str(j):0 for j in range(8)},"NODE_ARM_RECEIPT")
            observed=[]
            expected_marker=NETWORK[mode].encode()
            for j in range(8):
                rank,h=read(directory/f"card-{j}.json")
                receipts[f"{i}/{node}/{j}"]=h
                need(rank["step"]==step and rank["card"]==j and rank["node"]==node and not rank["error"],"ACTUAL_CARD_RECEIPT")
                observed.append(rank["pci"])
                r,=rank["results"]
                need(r["payload_bytes"]==26214400 and r["verified_elements"]==13107200
                     and r["logical_rank"]==side*4+j%4 and r["group"]==j//4 and r["network_policy"]==NETWORK[mode],
                     "MATCHED_PAYLOAD_AND_WORLD_SIZE")
                times=r["rank_max_seconds"]
                need(len(times)==96 and all(math.isfinite(x) and x>0 for x in times),"TIMED_SERIES")
                log=directory/f"card-{j}.log"
                need(0<log.stat().st_size<=512*1024,"LOG_BOUND")
                raw=log.read_bytes()
                need(set(re.findall(rb"Using network (Socket|IBext_v8)\b",raw))=={expected_marker} and
                     set(re.findall(rb"via NET/(Socket|IBext(?:_v8)?)(?:/|\b)",raw))=={expected_marker},"ACTUAL_DATA_CHANNEL")
                receipts[f"{i}/{node}/{j}-log"]=hashlib.sha256(raw).hexdigest()
                sample[node,j]=times
            need(sorted(observed)==pci[node] and len(set(observed))==8,"EIGHT_PHYSICAL_DEVICES")
        pair_series=[]
        for group in range(2):
            values=[v for (_,j),v in sample.items() if j//4==group]
            need(len(values)==8 and all(v==values[0] for v in values),"SLOWEST_RANK_AGREEMENT")
            pair_series.extend(values[0])
        medians.append(statistics.median(pair_series)*1000)
    socket=statistics.median(medians[0:4:3])
    shca=statistics.median(medians[1:3])
    return dict(nodes=pair,arms_median_ms=medians,socket_ms=socket,shca_ms=shca,socket_over_shca=socket/shca,
                receipt_index_SHA256=hashlib.sha256(json.dumps(receipts,sort_keys=True).encode()).hexdigest())
