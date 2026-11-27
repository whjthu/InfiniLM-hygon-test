"""Run one published CPU-only test of 96 actual InternVL inference inputs."""

import argparse
from base64 import b64encode
import json
from pathlib import Path
import re
import subprocess
from types import ModuleType
from uuid import uuid4

from run_hygon_bound_payload import digest, document, send


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOURCE = REPO / "code/bw1000/training-domain-pilot-v1"
MEMBERS = ("internvl_gqa_val96.py", "internvl_gqa_val96_native.py", "internvl_gqa_val96_cpu.py")
REQUEST = "e362a0729ae5483bbb1c3164623372cd"
LABEL = "training-internvl-gqa-val96-cpu-native-v2"
UPLOAD_PIN = "bd65e3b2cfe4779168ffb937e5aaf8559594c9f8d8307486cc031870f9f3ac46"
INPUT_PIN = "98f729fce9e6af91edb10bbf4dcc9b0d06f1322a38247c7001d05b02a21fc46b"


REMOTE = r'''
from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

base = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
results = base / "results"
book = results / "training-internvl-gqa-val96-user-only-v1"
run = results / LABEL
parent, node = "1103606", "f13r1n14"
host_python = "/usr/local/python3.12/bin/python3"
apptainer = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
cols = "JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"

def hash_bytes(raw):
    return sha256(raw).hexdigest()

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def store(path, value):
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hash_bytes(data)

def query(*argv):
    return subprocess.check_output(argv, timeout=8, text=True).splitlines()

def main():
    package = json.load(sys.stdin)
    selected = package["files"]
    require(set(selected) == set(MEMBERS) and re.fullmatch(r"[0-9a-f]{40}",package["publication"]),
            "Exact published CPU source required")
    ready = (book / "verified-upload.json").read_bytes()
    require(hash_bytes(ready) == UPLOAD_PIN and json.loads(ready)["input_SHA256"] == INPUT_PIN,
            "Actual user-only upload receipt required")
    [line] = query("scontrol","show","job","--oneliner",parent)
    fields = dict(part.split("=",1) for part in line.split() if "=" in part)
    require(fields["JobId"] == parent and fields["JobState"] == "RUNNING"
            and fields["UserId"].endswith(f"({os.getuid()})") and "gres/dcu=320" in fields["AllocTRES"].split(",")
            and node in query("scontrol","show","hostnames",fields["NodeList"]), "Current exact placement differs")
    files = {name: b64decode(raw, validate=True) for name, raw in selected.items()}
    for name, raw in files.items():
        require(0 < len(raw) < 32768 and Path(name).name == name, "Source member out of scope")
        compile(raw, name, "exec")
    runtime = results / "training-vlm-swift-offline-runtime-v1"
    framework = results / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
    metadata = results / "training-vlm-T03-T04-metadata-v2/internvl/model"
    image = base / "containers/dtk26-sandbox-r1"
    require(all(path.is_dir() for path in (runtime, framework, metadata, image)), "Original runtime roots unavailable")
    os.umask(0o077)
    run.mkdir(mode=0o700)
    source = run / "source"
    source.mkdir()
    for name, raw in files.items():
        with (source / name).open("xb") as fd:
            fd.write(raw)
    work = run / "work"
    work.mkdir()
    epoch = time.time()
    deadline = epoch + 195
    args = ["srun","--jobid="+parent,"--job-name="+LABEL,"--exact","--immediate=5","--cpu-bind=cores",
            "-N1","-n1","-w",node,"-c4","--mem=8192M","--gres=none","--time=00:03:00",
            "/usr/bin/timeout","--kill-after=5s","165s",apptainer,"exec","--contain","--rocm","--no-home","--pwd","/"]
    for bind in (f"{book}/input:/gqa-input:ro",f"{metadata}:/InternVL3_5-8B-HF:ro",f"{source}:/gqa-source:ro",
                 f"{work}:/gqa-out:rw",f"{framework}:/gqa-framework:ro",f"{runtime}:/swift-runtime:ro"):
        args.extend(("--bind",bind))
    shell = ("set -euo pipefail; source /opt/dtk/env.sh >&2; "
             "export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; "
             "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1; "
             "export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false; "
             "export HOME=/tmp TMPDIR=/tmp HF_HOME=/tmp/gqa-hf XDG_CACHE_HOME=/tmp/gqa-xdg "
             "PYTHONPATH=/gqa-framework:/swift-runtime/site; exec /usr/bin/python3 -B -u \"$@\"")
    args.extend([str(image),"/bin/bash","--noprofile","--norc","-c",shell,"gqa96-cpu",
                 "/gqa-source/internvl_gqa_val96_cpu.py","--inputs","/gqa-input","--metadata","/InternVL3_5-8B-HF",
                 "--output","/gqa-out/native.json","--parent",parent,"--node",node,"--deadline",str(deadline)])
    store(run/"intent.json",{"request":REQUEST,"stage":"CPU_ONLY","epoch":epoch,"deadline":deadline,
                            "target_args":args,"commit":package["publication"],
                            "sources":{name:hash_bytes(raw) for name, raw in files.items()}})
    with (run/"stdout.log").open("xb") as stdout, (run/"stderr.log").open("xb") as stderr:
        child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            state = Path("/proc",str(child.pid),"stat").read_text().rsplit(")",1)[-1].split()
            store(run/"owned-srun.json",{"pid":child.pid,"startticks":int(state[19]),"request":REQUEST})
        except FileNotFoundError:
            pass
        try:
            rc = child.wait(timeout=182)
        except subprocess.TimeoutExpired:
            rc = None
    manifest = work/"native.json"
    raw = manifest.read_bytes() if manifest.is_file() else None
    proof = json.loads(raw) if raw is not None else None
    account_for = proof["step"] if proof else parent
    require(re.fullmatch(parent + r"(?:\.\d+)?",account_for), "Unexpected CPU accounting handle")
    rows = []
    bound = time.monotonic() + 9
    while time.monotonic() < bound:
        rows = [line.split("|") for line in query("sacct","-nP","-j",account_for,"-o",cols)]
        rows = [row for row in rows if len(row) == 9 and re.fullmatch(parent + r"\.\d+", row[0]) and row[1] == LABEL]
        if len(rows) == 1 and rows[0][2].split()[0] in {
                "COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED"} and rows[0][8] != "Unknown":
            break
        time.sleep(1.0)
    no_dcu = (len(rows) == 1 and rows[0][5] == node and "gres/" not in rows[0][6])
    passed = (rc == 0 and no_dcu and rows[0][2:4] == ["COMPLETED","0:0"] and proof is not None
              and proof["step"] == rows[0][0] and proof["input_SHA256"] == INPUT_PIN
              and len(proof["rows"]) == 96 and proof["counts"] == {"query":48,"verify":48}
              and all(proof[key] is False for key in ("model_weights_loaded","GPU_initialized","gold_read")))
    tail = (run/"stderr.log").read_bytes()
    errors = [line.split(":",1)[0] for line in tail.decode("utf-8","replace").splitlines()[-12:]
              if re.fullmatch(r"[A-Za-z]+(?:Error|Exception|Warning)",line.split(":",1)[0])]
    value = {"request":REQUEST,"rc":0 if passed else 1,"srun_rc":rc,"accounting":rows,
             "input_SHA256":INPUT_PIN,"CPU_native_rows":len(proof["rows"]) if proof else 0,
             "native_receipt_SHA256":hash_bytes(raw) if raw else None,
             "input_fields":proof["input_fields"] if proof else None,
             "input_token_range":[min(row["input_tokens"] for row in proof["rows"]),
                                  max(row["input_tokens"] for row in proof["rows"])] if proof else None,
             "stderr_SHA256":hash_bytes(tail),"error_classes":errors,"GPU_GO":False}
    pin = store(run/"outcome.json",value)
    print(json.dumps({**value,"outcome_SHA256":pin},sort_keys=True),flush=True)

try:
    main()
except (ValueError, OSError, subprocess.SubprocessError, KeyError) as error:
    print(json.dumps({"request":REQUEST,"rc":1,"class":type(error).__name__,
                      "reason_SHA256":hash_bytes(str(error).encode())},sort_keys=True),flush=True)
    sys.exit(1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--publication")
    args = parser.parse_args()
    source = f"REQUEST={REQUEST!r}\nLABEL={LABEL!r}\nMEMBERS={MEMBERS!r}\n"
    source += f"UPLOAD_PIN={UPLOAD_PIN!r}\nINPUT_PIN={INPUT_PIN!r}\n" + REMOTE
    compile(source, "GQA96 CPU original HOST", "exec")
    bodies = {name: (SOURCE / name).read_bytes().replace(b"\r\n", b"\n") for name in MEMBERS}
    for name, raw in bodies.items():
        compile(raw, name, "exec")
    if args.check:
        print(json.dumps({"request": REQUEST, "host_SHA256": digest(source.encode()),
                          "members": {k:digest(v) for k,v in bodies.items()}, "local_check_only": True}))
        return 0
    if not re.fullmatch("[0-9a-f]{40}", args.publication):
        raise ValueError("Exact published CPU source revision required")
    for path in (Path(__file__).resolve(), HERE/"run_hygon_bound_payload.py", *(SOURCE/name for name in MEMBERS)):
        key = path.relative_to(REPO).as_posix()
        found = subprocess.check_output(["git", "-c", f"safe.directory={REPO.as_posix()}",
                                         "show", args.publication + ":" + key], cwd=REPO)
        if found != path.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("CPU publication bytes differ")
    local = REPO / "local-only"
    helper = local / "run-minicpm-sdpa-target-asset-preflight-original.py"
    raw = helper.read_bytes()
    if digest(raw) != "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("Existing project transport differs")
    transport = ModuleType("existing_ssh")
    exec(compile(raw, str(helper), "exec"), vars(transport))
    package = {"publication":args.publication, "files":{k:b64encode(v).decode() for k,v in bodies.items()}}
    result = send(transport.ssh(REPO.parent), request=REQUEST, program=source, data=document(package),
                  send_token=local / f"internvl-gqa96-cpu-{REQUEST}-send.json", outcome_seconds=224)
    receipt = local / f"internvl-gqa96-cpu-{REQUEST}-{uuid4().hex}.json"
    with receipt.open("xb") as stream:
        stream.write(document(result))
    response = result.get("response") or {}
    print(json.dumps({"request": REQUEST,"receipt":receipt.name,"remote_outcome":result["remote_outcome"],
                      "remote_received":result["remote_received"],"response":response},sort_keys=True))
    return int(result["remote_outcome"] != "RESPONSE_OBSERVED" or response.get("rc") != 0)


if __name__ == "__main__":
    raise SystemExit(main())
