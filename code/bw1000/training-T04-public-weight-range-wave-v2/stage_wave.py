"""Fetch pinned public ranges in memory and stage them over ordinary project SSH."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import runpy
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HELPER = Path("C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py")
HELPER_SHA = "3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1"


def need(valid, detail):
    if not valid:
        raise ValueError(detail)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, record):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")


def load(path):
    return json.loads(path.read_bytes())


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        url = urllib.parse.urlsplit(newurl)
        need(url.scheme == "https" and url.hostname == "us.aws.cdn.hf.co" and
             not url.username and not url.password and url.port in (None, 443), "UNEXPECTED_OFFICIAL_REDIRECT")
        return urllib.request.Request(newurl, headers={"Range": req.get_header("Range"), "Accept-Encoding": "identity"})


def read_range(plan, chunk, left):
    start = time.monotonic()
    length = chunk["length"]
    offset = chunk["offset"]
    end = offset + length - 1
    request = urllib.request.Request(
        f"https://huggingface.co/{plan['repository']}/resolve/{plan['revision']}/{plan['file']}",
        headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"})
    opener = urllib.request.build_opener(OfficialRedirect())
    with opener.open(request, timeout=min(18, left(30))) as r:
        need(r.status == 206 and r.headers.get("Content-Range") == f"bytes {offset}-{end}/{plan['shard_bytes']}"
             and r.headers.get("Content-Length") == str(length)
             and r.headers.get("Content-Encoding", "identity").lower() == "identity", "OFFICIAL_CONTENT_RANGE")
        payload = bytearray()
        while len(payload) < length:
            left(25)
            part = r.read(min(1024**2, length - len(payload)))
            if not part:
                raise EOFError("OFFICIAL_RANGE_TRUNCATED")
            payload.extend(part)
        need(not r.read(1), "OFFICIAL_RANGE_EXCESS")
    return payload, time.monotonic() - start


def source_gate(publication):
    git = ["git", "-c", "safe.directory=" + str(REPO)]
    def call(*args):
        return subprocess.check_output([*git,*args],cwd=REPO,timeout=12)
    rel = HERE.relative_to(REPO).as_posix()
    manifest = (HERE / "SOURCE_FILES.json").read_bytes()
    need(call("rev-parse", "HEAD").decode().strip() == publication and
         not call("diff", "--cached", "--name-only").strip() and
         not call("status", "--porcelain", "--", rel).strip() and
         manifest == call("cat-file", "blob", publication + ":" + rel + "/SOURCE_FILES.json"),
         "EXACT_SOURCE_PUBLICATION")
    for name, expected in json.loads(manifest)["files"].items():
        raw = (HERE / name).read_bytes()
        need(dict(bytes=len(raw), SHA256=sha(raw)) == expected, "EXACT_SOURCE_MEMBER")
    return sha(manifest)


def invoke(p, payload, seconds):
    need(sha(HELPER.read_bytes()) == HELPER_SHA,"EXACT_SSH_HELPER")
    argv = runpy.run_path(str(HELPER))["ssh"]
    source = (HERE / "endpoint.py").read_text(encoding="utf-8")
    boot = "PARAMETERS="+repr(p)+"\n"+source
    command = "/usr/local/python3.12/bin/python3 -I -S -B -c "+shlex.quote(boot)
    result = subprocess.run([argv[0],"-o","StrictHostKeyChecking=yes",*argv[1:],command],
                            cwd=REPO,input=payload,capture_output=True,timeout=seconds)
    if result.returncode != 0:
        return None, dict(rc=result.returncode, stderr_bytes=len(result.stderr), stderr_sha=sha(result.stderr))
    need(0 < len(result.stdout) <= 16*1024,"BOUNDED_REMOTE_RECEIPT")
    return json.loads(result.stdout), dict(rc=0)


def run(publication):
    manifest_sha = source_gate(publication)
    plan = load(HERE / "PLAN.json")
    chunks = plan["chunks"]
    need(len(chunks) == 4 and all(chunk == dict(offset=(i+1)*8388608,length=8388608)
                                  for i,chunk in enumerate(chunks)), "FROZEN_CONTIGUOUS_WAVE")
    local = REPO / "local-only" / HERE.name
    local.mkdir(mode=0o700)
    epoch, monotonic = time.time(), time.monotonic()
    deadline = epoch + plan["budget_seconds"]
    intent = dict(plan, publication=publication, epoch=epoch, deadline=deadline, manifest_sha=manifest_sha)
    save(local / "intent.json", intent)

    def left(reserve=0):
        seconds = min(deadline-time.time(),monotonic+plan["budget_seconds"]-time.monotonic())-reserve
        if seconds <= 0:
            raise TimeoutError("ORIGINAL_EPOCH_EXHAUSTED")
        return seconds

    shared = {key:intent[key] for key in ("target","epoch","deadline","publication","request","revision")}
    prepared, transport = invoke(dict(shared, mode="prepare", intent=intent), b"", min(22,left()))
    save(local / "prepare.json", dict(receipt=prepared,transport=transport))
    need(prepared == dict(request=plan["request"],original_prefix_verified=True,target_created=True),
         "TARGET_ADMISSION_RECONCILE_ORIGINAL")

    def one(chunk):
        tag = str(chunk["offset"])
        started = time.monotonic()
        record = dict(chunk, state="SOURCE_INCOMPLETE")
        try:
            payload, read_s = read_range(plan,chunk,left)
            p = dict(shared, mode="receive", chunk=chunk, source_sha=sha(payload))
            record.update(source_sha=p["source_sha"], source_HTTP206_seconds=read_s)
            save(local / (tag + ".source.json"), dict(p, source_HTTP206_seconds=read_s))
            record["state"] = "TARGET_UNKNOWN"
            send_started = time.monotonic()
            receipt, transport = invoke(p,payload,left(4))
            record.update(ssh_seconds=time.monotonic()-send_started, transport=transport)
            if receipt is not None:
                need(all(receipt[key] == value for key,value in p.items()) and
                     receipt["published_bytes"] == chunk["length"], "RECEIVER_IDENTITY")
                save(local / (tag + ".receiver.json"), receipt)
                record["state"] = "RECEIPT_RETURNED"
        except Exception as exc:
            record.update(error_type=type(exc).__name__, error_sha=sha(str(exc).encode()))
        record["elapsed_seconds"] = time.monotonic()-started
        save(local / (tag + ".outcome.json"), record)
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        ranges = list(pool.map(one,chunks))
    summary = dict(request=plan["request"], ranges=ranges, elapsed_seconds=time.monotonic()-monotonic,
                   data_saved_on_desktop_bytes=0)
    save(local / "outcome.json", summary)
    print(json.dumps(summary,allow_nan=False))
    return int(any(r["state"]!="RECEIPT_RETURNED" for r in ranges))


def check():
    local = REPO / "local-only" / HERE.name
    intent = load(local / "intent.json")
    deadline = time.time() + 35
    report, transport = invoke(dict(target=intent["target"], mode="audit", deadline=deadline,
                                    request=intent["request"], expected_intent=intent), b"", 40)
    need(report is not None and report["request"]==intent["request"], "AUDIT_ORIGINAL_SCOPE")
    for row in report["ranges"]:
        if row["state"]=="VERIFIED_RANGE":
            saved=load(local / (str(row["offset"]) + ".source.json"))
            need(saved["source_sha"]==row["source_sha"],"LOCAL_ORIGINAL_HTTP_RANGE_IDENTITY")
    save(local / "independent.json", dict(report=report,transport=transport))
    print(json.dumps(report,allow_nan=False))
    return int(sum(row["independently_verified_bytes"] for row in report["ranges"]) != 4*8388608)


if __name__ == "__main__":
    raise SystemExit(check() if sys.argv[1]=="--audit" else run(sys.argv[1]))
