"""Assemble the exact 20-file Z-Image runtime from verified public ranges."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import shutil
import signal
import ssl
import stat
import time
from urllib.request import HTTPSHandler, ProxyHandler, build_opener


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
ORIGINAL = PROJECT / "results/training-zimage-publisher-runtime-transfer-v1"
PREFETCH = PROJECT / "results/training-zimage-modelscope-independent-ranges-v1"


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def bound_file(path, size, pin=None):
    need(path.resolve(strict=True) == path, "Aliased source")
    with path.open("rb") as stream:
        raw = stream.read(size+1)
    need(len(raw) == size and (pin is None or sha256(raw).hexdigest() == pin), "Frozen byte binding differs")
    return raw


def opened(path):
    need(path.resolve(strict=True) == path, "Aliased data file")
    stream = os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb")
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        stream.close()
        raise ValueError("Expected regular input file")
    return stream


def sealed_copy(path, target, row, deadline):
    digest = sha256()
    seen = 0
    with opened(path) as source, target.open("xb") as fresh:
        need(os.fstat(source.fileno()).st_size == row["bytes"], "Sealed input size differs")
        while block := source.read(4 << 20):
            need(time.time() < deadline, "Copy deadline reached")
            need(fresh.write(block) == len(block), "Short sealed copy")
            digest.update(block)
            seen += len(block)
        need(seen == row["bytes"] and digest.hexdigest() == row["SHA256"], "Sealed input hash differs")
        fresh.flush()
        os.fsync(fresh.fileno())
    return seen


def prior_bytes(row, claimed_offset):
    directory = ORIGINAL / "segments" / f'{row["asset_id"]:02d}'
    with opened(directory / "checkpoint.json") as source:
        record = source.read(131073)
    need(0 < len(record) <= 131072, "Original checkpoint exceeds the allowed size")
    state = json.loads(record)
    expected = {"file_id": row["asset_id"], "filename": row["filename"], "segment_bytes": row["segment_bytes"],
                "bytes": row["bytes"], "SHA256": row["SHA256"]}
    need(state["spec"] == expected and state["confirmed_bytes"] == claimed_offset and state["sealed"] is False,
         "Original prefix changed after independent staging")
    with opened(directory / "payload.partial") as source:
        info = os.fstat(source.fileno())
        need(info.st_size == claimed_offset and info.st_ino == state["file"][1], "Original prefix file differs")
        offset = 0
        for receipt in state["segments"]:
            expected_bytes = min(row["bytes"]-offset, row["segment_bytes"])
            need(receipt["offset"] == offset and receipt["bytes"] == expected_bytes
                 and 0 < expected_bytes <= claimed_offset-offset, "Original prefix segment bounds differ")
            block = source.read(expected_bytes)
            need(len(block) == expected_bytes and sha256(block).hexdigest() == receipt["SHA256"],
                 "Original prefix segment digest differs")
            offset += expected_bytes
            yield block
        need(offset == claimed_offset and source.read(1) == b"", "Original prefix ledger does not close")


def indexed_prefetch(claim, receipt, row, offset):
    asset = row["asset_id"]
    directory = PREFETCH / f"asset-{asset}" / f"{offset:013d}"
    need(receipt["request"] == claim["prefetch_request"] and receipt["parent_SHA256"] == row["SHA256"]
         and receipt["revision"] == claim["publisher_revision"] and receipt["relative_path"] == row["relative_path"]
         and receipt["asset_id"] == asset and receipt["offset"] == offset and receipt["bytes"] == row["segment_bytes"],
         "Independent staged range differs from the frozen file")
    with opened(directory / "receipt.json") as stream:
        raw = stream.read(8193)
    need(0 < len(raw) <= 8192 and json.loads(raw) == receipt, "Independent staged receipt changed")
    return bound_file(directory / "range.bin", receipt["bytes"], receipt["SHA256"])


def download(root, temporary, row, claim, staged, direct, deadline, helpers):
    asset = str(row["asset_id"])
    requested_start = claim["prefix_offsets"].get(asset, 0)
    local_end = claim["prefetch_ends"].get(asset, 0)
    target = root / "candidate" / PurePosixPath(row["target_relative_path"])
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    need(target.parent.resolve(strict=True) == target.parent and not target.exists() and not target.is_symlink(),
         "Qualified destination path differs")
    progress = root / ("asset-" + asset + "-progress.jsonl")
    digest = sha256()
    total = 0
    network = 0
    reused = 0
    ca = ssl.create_default_context(cafile=direct["CA_FILE"])
    opener = build_opener(ProxyHandler({}), direct["PublisherRedirect"](), HTTPSHandler(context=ca))
    prior = prior_bytes(row, requested_start) if requested_start else iter(())
    started = time.monotonic()

    with temporary.open("xb") as destination, progress.open("x", encoding="utf8") as events:
        while total < row["bytes"]:
            need(time.time() < deadline-25, "Whole file window exhausted")
            count = min(row["segment_bytes"], row["bytes"]-total)
            offset = total
            if offset < requested_start:
                block = next(prior)
                source = "original_prefix_verified_by_range"
            elif offset < local_end:
                block = indexed_prefetch(claim, staged[(row["asset_id"], offset)], row, offset)
                source = "independent_stage_verified_by_range"
            else:
                block, _ = direct["fetch"](opener, row, offset, count, deadline-8)
                source = "official_fixed_revision_https"
            need(len(block) == count and destination.write(block) == count, "Short output range")
            digest.update(block)
            destination.flush()
            os.fsync(destination.fileno())
            total += count
            network += count if source == "official_fixed_revision_https" else 0
            reused += count if source != "official_fixed_revision_https" else 0
            events.write(json.dumps(dict(offset=offset, bytes=count, prefix=total, source=source, SHA256=sha256(block).hexdigest()))+"\n")
            events.flush()
            os.fsync(events.fileno())
        need(next(prior, None) is None, "Original prefix was not exhausted")
        need(digest.hexdigest() == row["SHA256"], "Whole-file publisher SHA-256 differs")
    temporary.rename(target)
    helpers["sync"](target.parent)
    helpers["sync"](root)
    return dict(asset_id=row["asset_id"], bytes=total, SHA256=row["SHA256"], network_bytes=network,
                reused_bytes=reused, wall_s=time.monotonic()-started)


def verified_path(root, row):
    path = PurePosixPath(row["target_relative_path"])
    need(not path.is_absolute() and ".." not in path.parts and path.name == row["filename"], "Unsafe publisher path")
    return root.joinpath(*path.parts)


def main():
    import resource

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("epoch", type=float)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    claim = json.loads((root / "CLAIM.json").read_bytes())
    need(root.name == claim["label"] and claim["request"] == args.request
         and 0 <= time.time()-args.epoch < 12 and "SLURM_JOB_ID" not in os.environ, "Invalid bounded owner")
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    deadline = args.epoch + claim["worker_seconds"]
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, deadline-time.time())
    started = time.monotonic()
    for name, meta in claim["providers"].items():
        bound_file(root / name, meta["bytes"], meta["SHA256"])
    direct = runpy.run_path(str(root / "frozen_direct.py"), run_name="frozen_direct")
    helpers = runpy.run_path(str(root / "range_helpers.py"), run_name="range_helpers")
    need(direct["REVISION"] == claim["publisher_revision"], "Publisher release differs")
    source_raw = bound_file(PREFETCH / "worker-result.json", claim["prefetch_receipt"]["bytes"], claim["prefetch_receipt"]["SHA256"])
    stage = json.loads(source_raw)
    need(stage["request"] == claim["prefetch_request"] and stage["errors"] == {} and stage["verified_range_count"] == 16,
         "Expected the original independently read back staging scope")
    staged = {(row["asset_id"], row["offset"]):row for row in stage["committed"]}
    rows = json.loads((root / "RUNTIME_FETCH_MANIFEST.json").read_bytes())["assets"]
    need(len(rows)==20 and sum(row["bytes"] for row in rows)==claim["runtime_expected_bytes"]
         and len({verified_path(root/"candidate", row) for row in rows})==20, "Runtime member denominator differs")
    need(shutil.disk_usage(root).free >= claim["runtime_expected_bytes"] + (1 << 30), "Target lacks capacity")
    completed, errors = [], {}
    (root / "candidate").mkdir(mode=0o700)
    helpers["write"](root/"worker-started.json",helpers["serial"](dict(request=args.request, epoch=args.epoch, deadline=deadline, pid=os.getpid())))
    downloads = set(claim["download_assets"])
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(download, root, root / f'{row["asset_id"]:02d}.partial', row, claim, staged, direct, deadline, helpers):row["asset_id"]
                for row in sorted(rows,key=lambda item:-item["bytes"]) if row["asset_id"] in downloads}
        for future in as_completed(jobs):
            try:
                completed.append(future.result())
            except Exception as error:
                errors[str(jobs[future])] = dict(type=type(error).__name__, code=getattr(error,"code",None),
                    detail=str(error)[:120] if type(error) in (ValueError,TimeoutError) else None)

    if not errors:
        for row in rows:
            if row["asset_id"] in downloads:
                continue
            target = verified_path(root/"candidate",row)
            target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            size = sealed_copy(verified_path(ORIGINAL/"candidate",row), target, row, deadline)
            helpers["sync"](target.parent)
            completed.append(dict(asset_id=row["asset_id"],bytes=size,SHA256=row["SHA256"],network_bytes=0,reused_bytes=size))
        need(len(completed) == 20 and len({x["asset_id"] for x in completed}) == 20, "Qualified file set is incomplete")
        (root/"candidate").rename(root/"runtime")
        helpers["sync"](root)
    report = dict(request=args.request, elapsed_s=time.monotonic()-started, errors=errors,
                  files=sorted(completed,key=lambda row:row["asset_id"]),
                  matched_whole_files=len(completed),actual_DCU=0,training_ready=False,
                  whole_runtime_published=len(completed)==20 and not errors,
                  network_bytes=sum(item["network_bytes"] for item in completed))
    helpers["write"](root/"worker-result.json",helpers["serial"](report))
    helpers["sync"](root)
    signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({key:report[key] for key in ("request","elapsed_s","actual_DCU","matched_whole_files",
                                                 "whole_runtime_published","network_bytes","errors")}),flush=True)
    return int(not report["whole_runtime_published"])


if __name__ == "__main__":
    raise SystemExit(main())
