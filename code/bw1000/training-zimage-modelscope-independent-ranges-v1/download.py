"""Fetch exact Z-Image ranges into a new stage without modifying original checkpoints."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import ssl
import time
from urllib.request import HTTPSHandler, ProxyHandler, build_opener


def need(condition, message):
    if not condition:
        raise ValueError(message)


def serial(doc):
    return (json.dumps(doc, sort_keys=True, allow_nan=False) + "\n").encode()


def write(path, raw):
    with path.open("xb") as stream:
        need(stream.write(raw) == len(raw), "Short range or receipt write")
        stream.flush()
        os.fsync(stream.fileno())


def sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def commit(folder, ident, data, network_seconds):
    need(len(data) == ident["bytes"] and 0 <= network_seconds, "Range length or elapsed time differs")
    label = f'{ident["offset"]:013d}'
    final = folder / label
    pending = folder / (label + ".part")
    need(not final.exists() and not final.is_symlink(), "Original range already committed")
    pending.mkdir(mode=0o700)
    receipt = dict(ident, SHA256=sha256(data).hexdigest(), network_seconds=network_seconds)
    write(pending / "range.bin", data)
    write(pending / "receipt.json", serial(receipt))
    sync(pending)
    pending.rename(final)
    sync(folder)
    return receipt


def stage_ranges(root, claim, direct, deadline):
    manifest_raw = (root / "RUNTIME_FETCH_MANIFEST.json").read_bytes()
    need(sha256(manifest_raw).hexdigest() == claim["runtime_manifest_SHA256"], "Frozen runtime manifest differs")
    assets = {str(asset["asset_id"]): asset for asset in json.loads(manifest_raw)["assets"]}
    completed = []
    context = ssl.create_default_context(cafile=direct["CA_FILE"])
    need(shutil.disk_usage(root).free >= claim["planned_bytes"] + (1 << 30), "Available stage capacity too small")

    def shard(item):
        asset, start = item
        row = assets[asset]
        chunk = row["segment_bytes"]
        need(chunk == 32 << 20 and start % chunk == 0 and start >= 0, "Range start differs from the frozen asset")
        need(start + chunk * claim["ranges_per_asset"] <= row["bytes"], "Ranges extend beyond the frozen asset")
        target = root / ("asset-" + asset)
        target.mkdir(mode=0o700)
        sync(root)
        opener = build_opener(ProxyHandler({}), direct["PublisherRedirect"](), HTTPSHandler(context=context))
        for idx in range(claim["ranges_per_asset"]):
            if time.time() >= deadline-20:
                break
            offset = start + idx * chunk
            data, elapsed = direct["fetch"](opener, row, offset, chunk, deadline-4)
            ident = dict(request=claim["request"], asset_id=int(asset), offset=offset, bytes=chunk,
                         revision=direct["REVISION"], parent_SHA256=row["SHA256"],
                         relative_path=row["relative_path"])
            completed.append(commit(target, ident, data, elapsed))

    errors = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {asset: pool.submit(shard, (asset, start)) for asset, start in claim["start_offsets"].items()}
        for asset, job in jobs.items():
            try:
                job.result()
            except Exception as error:
                errors[asset] = dict(type=type(error).__name__, code=getattr(error, "code", None),
                                     detail=str(error)[:110] if type(error) in (ValueError, TimeoutError) else None)
    completed.sort(key=lambda row:(row["asset_id"],row["offset"]))
    total = sum(row["bytes"] for row in completed)
    return dict(request=claim["request"], committed=completed, verified_range_count=len(completed),
                committed_bytes=total, planned_bytes=claim["planned_bytes"], errors=errors,
                full_runtime_qualified=False, actual_DCU=0, training_ready=False)


def main():
    import resource

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("epoch", type=float)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    claim = json.loads((root / "CLAIM.json").read_bytes())
    need(root.name == claim["label"] and args.request == claim["request"] and 0 <= time.time()-args.epoch < 10,
         "Expected one original start in its own namespace")
    need("SLURM_JOB_ID" not in os.environ, "Expected an unallocated foreground data staging process")
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    direct_file = root / "frozen_direct.py"
    direct_raw = direct_file.read_bytes()
    need(sha256(direct_raw).hexdigest() == claim["frozen_downloader_SHA256"], "Frozen publisher transport differs")
    direct = runpy.run_path(str(direct_file), run_name="frozen_publisher")
    need(direct["REVISION"] == claim["publisher_revision"], "Publisher revision differs")
    deadline = args.epoch + claim["worker_seconds"]
    signal.signal(signal.SIGALRM, lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, deadline-time.time())
    started = time.monotonic()
    write(root / "worker-started.json", serial(dict(request=args.request, epoch=args.epoch, deadline=deadline,
                                                       pid=os.getpid())))
    result = stage_ranges(root, claim, direct, deadline)
    result["elapsed_seconds"] = time.monotonic()-started
    write(root / "worker-result.json", serial(result))
    sync(root)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({key:result[key] for key in ("request","committed_bytes","planned_bytes","verified_range_count",
                                                  "actual_DCU","errors","elapsed_seconds")}), flush=True)
    return int(result["errors"] != {} or result["committed_bytes"] != result["planned_bytes"])


if __name__ == "__main__":
    raise SystemExit(main())
