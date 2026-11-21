"""Finish the single frozen sitting_xyz RGB-D pair using its one outstanding MCAP range."""

import hashlib
import json
from pathlib import Path
import re
import runpy
from time import monotonic
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIOR = REPO / "local-only/training-vggt-tum-sitting-xyz-network-recovery-v2"
OUTPUT = REPO / "local-only/training-vggt-tum-sitting-xyz-missing-depth-v1"
HELPER_SHA = "3bcb569c44ab2a33d7b53f843080c765706e2c67c79166df6ac96255214da3e0"
PRIOR_RESULT_SHA = "63138fdf11d0edbb17de1e21491304a8e1aa60fc3cb3dcd6895c96350ef78c07"
PRIOR_SOURCE_SHA = "a6308e1d87e1adbd40cf0cb0c084dc4a59b8e1212693d51510eb02cf9f310476"
RANGE_SHA = "bcd501287f757e732ed82f86a08b953f4568cdc92037d4277338c33e29f7fb10"
GEOMETRY_SHA = "c453a9c561c6411b8c2f25652f5c004c928db423c03ca0a45eba7d47ed9e3962"
RANGE_BYTES = 659648
SEQUENCE = "freiburg3_sitting_xyz"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_pinned(path, expected, limit):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit or digest(raw) != expected:
        raise ValueError(f"Missing or changed pinned artifact {path.name}")
    return raw


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def saved_requests():
    source = json.loads(read_pinned(PRIOR / "SOURCE.json", PRIOR_SOURCE_SHA, 16000))
    result = json.loads(read_pinned(PRIOR / "result.json", PRIOR_RESULT_SHA, 16000))
    saved = {}
    if len(source["retrievals"]) != 20 or len(result["requests"]) != 20:
        raise ValueError("Expected 20 prior completed bounded responses")
    for i, (details, record) in enumerate(zip(source["retrievals"], result["requests"]), start=1):
        name = f"HTTP-{i:02d}.private.bin"
        key = details["url"], record["range"]
        if (record["saved_name"] != name or record["SHA256"] != details["SHA256"]
                or record["URL_SHA256"] != digest(key[0].encode()) or record["read_bytes"] != details["bytes"]
                or record["status"] != (206 if key[1] else 200) or key in saved):
            raise ValueError("Prior response identity differs")
        saved[key] = (name, details)
    return saved, source["mirror_published_object_bytes"]


class MirrorRedirect(HTTPRedirectHandler):
    max_repeats = 1
    max_redirections = 3

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        host = target.hostname or ""
        if target.scheme != "https" or not (host == "huggingface.co"
                                             or host.endswith((".huggingface.co", ".hf.co"))):
            raise ValueError("Redirect outside verified public mirror")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_range(opener, url, start, total, deadline, attempt):
    end = start + RANGE_BYTES - 1
    byte_range = f"bytes={start}-{end}"
    req = Request(url, headers={"Range": byte_range, "Accept-Encoding": "identity"})
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("Range deadline elapsed before request")
    with opener.open(req, timeout=min(18, remaining)) as response:
        attempt.update(status=response.status)
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
            raise ValueError("Mirror did not provide the one pinned range")
        length = response.headers.get("Content-Length")
        if length is not None and length != str(RANGE_BYTES):
            raise ValueError("Range Content-Length differs")
        body = bytearray()
        while not response.isclosed() and len(body) <= RANGE_BYTES:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Range deadline elapsed during read")
            response.fp.raw._sock.settimeout(min(18, remaining))
            chunk = response.read(min(65536, RANGE_BYTES + 1 - len(body)))
            if not chunk:
                break
            body.extend(chunk)
            attempt["read_bytes"] = len(body)
        if len(body) != RANGE_BYTES:
            raise ValueError("Mirror range body length differs")
    raw = bytes(body)
    info = dict(url=url, bytes=len(raw), SHA256=digest(raw), start=start, end=end, total=total)
    attempt["SHA256"] = info["SHA256"]
    return raw, info


def main():
    saved, total = saved_requests()
    helper = HERE / "frozen_pair_acquisition.py"
    read_pinned(helper, HELPER_SHA, 15000)
    helper_scope = runpy.run_path(str(helper))["acquire"].__globals__
    mirror = helper_scope["HF"] + "resolve/" + helper_scope["REV"] + "/data/freiburg3/fr3_sitting_xyz.fo.mcap"
    archive = "rgbd_dataset_" + SEQUENCE
    helper_scope.update(SEQUENCE=SEQUENCE, ARCHIVE_ROOT=archive, MCAP=mirror, R=OUTPUT,
                        MREL="data/freiburg3/fr3_sitting_xyz.fo.mcap",
                        TUM="https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg3/" + archive)
    scorer = REPO / "code/bw1000/training-domain-pilot-v1/tum_vggt_pair.py"
    read_pinned(scorer, GEOMETRY_SHA, 18000)
    prepare_pair = runpy.run_path(str(scorer))["prepare_pair"]
    output = dict(sequence=SEQUENCE, planned_pairs=1, qualified_source_pairs=0, cached_responses=0,
                  fresh_attempts=[], GPU_steps=0, training_updates=0, READY=False, formal_CO3D=False)
    start_time = monotonic()
    OUTPUT.mkdir()
    write_json(OUTPUT / "intent.json", dict(range_plan_SHA256=RANGE_SHA, bytes=RANGE_BYTES,
                                          prior_result_SHA256=PRIOR_RESULT_SHA, prior_source_SHA256=PRIOR_SOURCE_SHA))
    consumed = set()
    opener = build_opener(MirrorRedirect())

    def request(url, cap, byte_range=None):
        key = (url, byte_range)
        if key in saved:
            if key in consumed:
                raise ValueError("Repeated local response")
            name, info = saved[key]
            raw = read_pinned(PRIOR / name, info["SHA256"], cap)
            if len(raw) != info["bytes"]:
                raise ValueError("Cached byte count differs")
            consumed.add(key)
            output["cached_responses"] = len(consumed)
            return raw, info
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", byte_range or "")
        if (len(consumed) != len(saved) or output["fresh_attempts"] or url != mirror + "?download=true"
                or match is None or cap < RANGE_BYTES):
            raise ValueError("No other network request is in this recovery scope")
        start, end = map(int, match.groups())
        observed_proof = digest(json.dumps([(start, end - start + 1)], separators=(",", ":")).encode())
        if end >= total or end - start + 1 != RANGE_BYTES or observed_proof != RANGE_SHA:
            raise ValueError("Depth range differs from exact plan")
        attempt = dict(range_plan_SHA256=RANGE_SHA, read_bytes=0, status=None)
        output["fresh_attempts"].append(attempt)
        write_json(OUTPUT / "request-intent.json", attempt)
        raw, info = fetch_range(opener, url, start, total, start_time + 55, attempt)
        (OUTPUT / "depth-chunk.private.bin").write_bytes(raw)
        return raw, info

    helper_scope["request"] = request
    try:
        helper_scope["acquire"]()
        if len(consumed) != 20 or len(output["fresh_attempts"]) != 1:
            raise ValueError("Acquisition did not consume the exact replay and range plan")
        manifest_raw = (OUTPUT / "pair-manifest.json").read_bytes()
        _, evidence = prepare_pair(OUTPUT, json.loads(manifest_raw))
        write_json(OUTPUT / "geometry.private.json", evidence)
        output.update(state="SOURCE_AND_BIDIRECTIONAL_GEOMETRY_QUALIFIED", qualified_source_pairs=1,
                      manifest_SHA256=digest(manifest_raw), baseline_m=evidence["baseline_m"],
                      valid_depth_pixels=evidence["valid_depth_pixels"],
                      depth_reprojection=evidence["depth_reprojection"])
    except Exception as error:
        output.update(state="SOURCE_OR_GEOMETRY_INCOMPLETE", failure_type=type(error).__name__)
        write_json(OUTPUT / "error.private.json", dict(type=type(error).__name__, message=str(error)[:384]))
    output["wall_seconds"] = monotonic() - start_time
    write_json(OUTPUT / "result.json", output)
    print(json.dumps(output, allow_nan=False))
    return 0 if output["qualified_source_pairs"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
