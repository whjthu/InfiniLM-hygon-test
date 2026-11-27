"""Independently verify the bounded IPC event stream and closure status."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


def need(ok,error):
    if not ok:
        raise ValueError(error)


def raw(path,cap):
    need(path.is_file() and not path.is_symlink(),"REGULAR_SIDECAR_ONLY")
    with path.open("rb") as stream:
        value=stream.read(cap+1)
    need(len(value)<=cap,"SIDECAR_EXTENT")
    return value


def review(folder,request,summary_sha,retirement_sha):
    life=raw(folder/"short-private-tmp-lifecycle.json",256<<10)
    retirement=raw(folder/"short-private-tmp-cleanup.json",2<<20)
    need(sha256(life).hexdigest()==summary_sha and sha256(retirement).hexdigest()==retirement_sha,
         "INDEPENDENT_SIDECAR_SHA")
    l,r=map(json.loads,(life,retirement))
    need(l["request_id"]==r["request_id"]==request and
         l["schema"]=="GEMMA_R120_REQUEST_SHORT_DIRECTORY_LIFECYCLE_V2","BOUND_IPC_OWNER_REQUEST")
    a=l["event_archive"]
    need(a["relative"]=="short-private-tmp-events.jsonl","EXACT_FIXED_EVENT_SIDECAR")
    body=raw(folder/a["relative"],4<<20)
    lines=body.splitlines(keepends=True)
    need(a["bytes"]==len(body) and a["SHA256"]==sha256(body).hexdigest()
         and a["count"]==len(lines)<=16384,"COMPLETE_JOURNAL_HASH_AND_COUNT")
    first=[]
    for seq,line in enumerate(lines,1):
        row=json.loads(line)
        need(line.endswith(b"\n") and row.pop("sequence")==seq and row["event"] in {"ADDED","DISAPPEARED","CHANGED"}
             and re.fullmatch("[0-9a-f]{64}",row["basename_SHA256"])
             and set(row)=={"stage","elapsed_ms","event","basename_SHA256","current","previous"},"JOURNAL_EVENT_SCHEMA_ORDER")
        if seq<=128:
            first.append(row)
    need(first==l["events"] and l["inline_omitted_events"]==len(lines)-len(first),"INLINE_AND_SIDECAR_PROJECTION")
    cleanup=r["retirement"]
    need(cleanup["state"] in {"MARKER_ONLY","REFUSED"} and cleanup["removed"]==0,"NATURAL_RETIREMENT_CUSTODY")
    return dict(request=request,captured_events=a["count"],inline_omitted_events=l["inline_omitted_events"],
                actual_dropped_events=l["truncated_events"],observation_errors=len(l["errors"]),
                marker_only_natural=cleanup["state"]=="MARKER_ONLY",retained_residue=len((cleanup["before"] or {}).get("residuals",{})),
                cleanup_reason=cleanup.get("refusal",{}).get("code"),
                passed=cleanup["state"]=="MARKER_ONLY" and l["truncated_events"]==0 and not l["errors"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--request",required=True)
    parser.add_argument("--lifecycle-sha256",required=True)
    parser.add_argument("--cleanup-sha256",required=True)
    arg=parser.parse_args()
    result=review(arg.directory,arg.request,arg.lifecycle_sha256,arg.cleanup_sha256)
    print(json.dumps(result,sort_keys=True))
    raise SystemExit(int(not result["passed"]))
