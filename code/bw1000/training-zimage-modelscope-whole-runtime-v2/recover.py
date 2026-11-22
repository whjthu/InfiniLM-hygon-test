"""Complete only missing publisher assets after an independently observed terminal run."""

from argparse import ArgumentParser
from hashlib import sha256
from http.client import IncompleteRead, RemoteDisconnected
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, ProxyHandler, build_opener


RUNS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
ORIGINAL = RUNS / "training-zimage-publisher-runtime-transfer-v1"
PREVIOUS = RUNS / "training-zimage-modelscope-whole-runtime-v1"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_bounded(path, maximum):
    need(path.resolve(strict=True) == path, "Aliased evidence")
    with path.open("rb") as stream:
        data = stream.read(maximum+1)
    need(0 < len(data) <= maximum, "Evidence exceeded its registered bounds")
    return data


def retry_range(fetch, opener, row, offset, count, until, pause):
    attempts = []
    for index in range(3):
        try:
            data, seconds = fetch(opener, row, offset, count, until)
            return data, seconds, attempts
        except (HTTPError, URLError, TimeoutError, ConnectionResetError, IncompleteRead, RemoteDisconnected) as error:
            code = getattr(error, "code", None)
            retry = (code in (408, 425, 429, 500, 502, 503, 504) if isinstance(error, HTTPError)
                     else True)
            delay = 2 << index
            if not retry or index == 2 or time.time() + delay + 16 >= until:
                raise
            attempts.append(dict(type=type(error).__name__,code=code))
            pause(delay)
    raise RuntimeError("Original retry bound exceeded")


def repeat_fetch(fetch, opener, row, offset, count, until):
    try:
        return retry_range(fetch, opener, row, offset, count, until, time.sleep)
    except ValueError as error:
        if str(error) != "Publisher range ended early" or count <= 8 << 20 or time.time()+45 >= until:
            raise
    combined = bytearray()
    elapsed = 0
    attempts = [dict(type="ShortPublisherBody", code=None)]
    while len(combined) < count:
        length = min(8 << 20, count-len(combined))
        block, seconds, further = retry_range(fetch, opener, row, offset+len(combined), length, until, time.sleep)
        need(len(block) == length, "Repartitioned publisher span differs")
        combined.extend(block)
        elapsed += seconds
        attempts.extend(further)
    return combined, elapsed, attempts


def archived_prefix(path, ledger, row, expected, opened):
    offset = 0
    with opened(path) as source:
        need(os.fstat(source.fileno()).st_size == expected, "Previous partial has an unaccounted extent")
        for entry in ledger:
            count = min(row["segment_bytes"], row["bytes"]-offset)
            need(entry["offset"] == offset and entry["prefix"] == offset+count
                 and entry["bytes"] == count and offset+count <= expected, "Discontinuous previous durable prefix")
            block = source.read(count)
            need(len(block)==count and sha256(block).hexdigest()==entry["SHA256"], "Previous range digest differs")
            offset+=count
            yield block
        need(offset==expected and source.read(1)==b"", "Previous prefix did not close")


def link_verified(source, target, row, opened):
    fingerprint = sha256()
    with opened(source) as stream:
        info=os.fstat(stream.fileno())
        need(info.st_size==row["bytes"], "Previously qualified size differs")
        while block:=stream.read(4 << 20):
            fingerprint.update(block)
        need(fingerprint.hexdigest()==row["SHA256"] and os.path.samestat(info, source.lstat()),
             "Previously qualified full digest or inode differs")
        os.link(source,target,follow_symlinks=False)
        need(not target.is_symlink() and os.path.samestat(info,target.lstat()), "Qualified copy link differs")


def stage_missing(root, claim, row, old, direct, helpers, deadline):
    asset = str(row["asset_id"])
    prior = claim["previous_prefix"]
    ledger_raw = read_bounded(PREVIOUS / ("asset-" + asset + "-progress.jsonl"), 262144)
    need(sha256(ledger_raw).hexdigest()==claim["previous_progress_SHA256"] and ledger_raw.endswith(b"\n"),
         "Previous progress changed since terminal audit")
    ledger=[json.loads(line) for line in ledger_raw.splitlines()]
    need(len(ledger)==claim["previous_progress_segments"], "Previous progress denominator differs")
    pieces=archived_prefix(PREVIOUS / (asset.zfill(2)+".partial"),ledger,row,prior,old["opened"])
    opener=build_opener(ProxyHandler({}),direct["PublisherRedirect"](),
                        HTTPSHandler(context=ssl.create_default_context(cafile=direct["CA_FILE"])))
    result_file=root/(asset.zfill(2)+".partial")
    dest=old["verified_path"](root/"candidate",row)
    dest.parent.mkdir(parents=True,exist_ok=True)
    digest=sha256()
    offset=0
    retries=0
    with result_file.open("xb") as output, (root/"missing-file-progress.jsonl").open("x",encoding="utf8") as events:
        while offset < row["bytes"]:
            need(time.time()<deadline-20, "Bounded publisher recovery window expired")
            count=min(row["segment_bytes"],row["bytes"]-offset)
            if offset < prior:
                block=next(pieces)
                source="previous_terminal_verified_range"
                attempts=[]
            else:
                block,_,attempts=repeat_fetch(direct["fetch"],opener,row,offset,count,deadline-5)
                source="official_fixed_revision_https"
            need(len(block)==count and output.write(block)==count, "Short output from bounded recovery")
            retries+=len(attempts)
            digest.update(block)
            output.flush()
            os.fsync(output.fileno())
            events.write(json.dumps(dict(offset=offset, bytes=count, prefix=offset+count, source=source,
                                         SHA256=sha256(block).hexdigest(), retry_errors=attempts))+"\n")
            events.flush()
            os.fsync(events.fileno())
            offset+=count
        need(next(pieces,None) is None and digest.hexdigest()==row["SHA256"], "Whole missing file digest differs")
    result_file.rename(dest)
    helpers["sync"](dest.parent)
    return dict(asset_id=row["asset_id"],bytes=offset,SHA256=row["SHA256"],
                network_bytes=offset-prior,reused_bytes=prior,retry_count=retries)


def main():
    import resource

    parser=ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("epoch",type=float)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    claim=json.loads((root/"CLAIM.json").read_bytes())
    need(root.name==claim["label"] and args.request==claim["request"] and
         "SLURM_JOB_ID" not in os.environ and 0<=time.time()-args.epoch<12,"Expected one unallocated foreground worker")
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    deadline=args.epoch+claim["worker_seconds"]
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
    for path,entry in claim["providers"].items():
        raw=read_bounded(root/path,entry["bytes"])
        need(len(raw)==entry["bytes"] and sha256(raw).hexdigest()==entry["SHA256"],"Frozen provider differs")
    old=runpy.run_path(str(root/"prior_assembler.py"),run_name="original_assembler")
    direct=runpy.run_path(str(root/"frozen_direct.py"),run_name="frozen_direct")
    helpers=runpy.run_path(str(root/"range_helpers.py"),run_name="range_helpers")
    need(direct["REVISION"]==claim["publisher_revision"],"Publisher revision changed")
    raw=read_bounded(PREVIOUS/"worker-result.json",claim["previous_worker"]["bytes"])
    need(len(raw)==claim["previous_worker"]["bytes"] and sha256(raw).hexdigest()==claim["previous_worker"]["SHA256"],
         "Original terminal receipt changed")
    prev=json.loads(raw)
    staged={row["asset_id"]:row for row in prev["files"]}
    missing=claim["missing_asset"]
    need(prev["request"]==claim["previous_request"] and prev["whole_runtime_published"] is False and
         {int(key) for key in prev["errors"]}=={missing} and set(staged)=={5,6,14,16},
         "Original terminal asset set differs")
    rows=json.loads((root/"RUNTIME_FETCH_MANIFEST.json").read_bytes())["assets"]
    need(len(rows)==20 and sum(row["bytes"] for row in rows)==claim["total_runtime_bytes"],"Complete expected file set differs")
    by_id={row["asset_id"]:row for row in rows}
    capacity=by_id[missing]["bytes"]+sum(row["bytes"] for row in rows if row["asset_id"] not in staged and row["asset_id"]!=missing)
    need(shutil.disk_usage(root).free>=capacity+(1<<30),"Insufficient new target capacity")
    (root/"candidate").mkdir(mode=0o700)
    helpers["write"](root/"worker-started.json",helpers["serial"](dict(request=args.request,epoch=args.epoch,deadline=deadline,pid=os.getpid())))
    completed=[]
    errors={}
    try:
        completed.append(stage_missing(root,claim,by_id[missing],old,direct,helpers,deadline))
        for row in rows:
            asset=row["asset_id"]
            if asset==missing:
                continue
            target=old["verified_path"](root/"candidate",row)
            target.parent.mkdir(parents=True,exist_ok=True)
            if asset in staged:
                entry=staged[asset]
                need((entry["bytes"],entry["SHA256"])==(row["bytes"],row["SHA256"]),"Original completed asset differs")
                link_verified(old["verified_path"](PREVIOUS/"candidate",row),target,row,old["opened"])
            else:
                old["sealed_copy"](old["verified_path"](ORIGINAL/"candidate",row),target,row,deadline)
            helpers["sync"](target.parent)
            completed.append(dict(asset_id=asset,bytes=row["bytes"],SHA256=row["SHA256"],network_bytes=0))
        need({item["asset_id"] for item in completed}==set(by_id) and len(completed)==20,"Incomplete runtime")
        (root/"candidate").rename(root/"runtime")
        helpers["sync"](root)
    except Exception as error:
        errors=dict(type=type(error).__name__,code=getattr(error,"code",None),
                    detail=str(error)[:120] if isinstance(error,ValueError) else None)
    report=dict(request=args.request,completed=completed,matched_whole_files=len(completed),
                whole_runtime_published=len(completed)==20 and not errors,errors=errors,actual_DCU=0,training_ready=False)
    helpers["write"](root/"worker-result.json",helpers["serial"](report))
    helpers["sync"](root)
    signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({key:report[key] for key in ("request","matched_whole_files","whole_runtime_published","errors","actual_DCU")}),flush=True)
    return int(not report["whole_runtime_published"])


if __name__=="__main__":
    raise SystemExit(main())
