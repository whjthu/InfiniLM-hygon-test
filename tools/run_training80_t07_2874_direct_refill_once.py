from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import runpy
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
PRIVATE = REPO / "local-only"
WORK = REPO.parent
NODE = "f13r2n07"
RUNNER = WORK / "run-dcu-standby-cycle-20261123.ps1"
PREVIOUS = "3fb996b1b2614717b2a90e18f3f4350c"
MARKER = PRIVATE / "training80-T07-predecessor2874-direct-guarded-once.json"
READER = PRIVATE / "read-training80-T07-original2874-dual-source.py"
READER_PIN = "e6590cefdfc3a9cb4f44980a9796847a144690bb7468cfea2f4ea4ae8d0aa2d2"


def digest(data):
    return sha256(data).hexdigest()


def save(path, data):
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return digest(data)


def json_data(doc):
    return (json.dumps(doc, sort_keys=True, allow_nan=False) + "\n").encode()


def need(cond, why):
    if not cond:
        raise ValueError(why)


def prior_proof(proof, pin, *, preparation_only=False):
    sys.path.insert(0, str(REPO/"tools"))
    import report_training80_backlog as mod
    need(proof.parent == PRIVATE and proof.name.startswith("training80-f13r2n07-original2874-predecessor-current-"),
         "not the T07 fresh terminal evidence")
    raw = proof.read_bytes()
    need(digest(raw) == pin and digest(RUNNER.read_bytes()) == "04f2bef937f98b35a29c0bfce7c199e56e049e001a0a5b92ecbf12878a358c73",
         "proof or qualified direct runner bytes differ")
    proof_record = json.loads(raw)
    obs = proof_record["observed"]
    target = obs
    need(proof_record["rc"] == 0 and obs["owner"] and obs["alloc_320"] and obs["parent_state"] == "RUNNING"
         and target["request"] == PREVIOUS
         and target["expected"]["SHA256"] == "16b5ea811c93e0b356f9c7f6952ffd58bd6627c00ef1236b6342f19bc16b4e50"
         and target["worker"]["SHA256"] == "8307ed89a04a69400d74ea1a6792875d1a3c7929bc671558a726cf883bc8b353",
         "T07 parent or original worker evidence differs")
    gpu, = target["accounting"]["1103606.2874"]
    probe, = target["accounting"]["1103606.2873"]
    need(probe[:4] == ["1103606.2873","training80-admit-f337e6d75e920a8427f50700adaee591","COMPLETED","0:0"]
         and probe[5:7] == [NODE,"cpu=1,mem=256M,node=1"]
         and probe[8] != "Unknown", "T07 preserved zero-GRES admission differs")
    expected = target["expected"]["document"]
    worker = target["worker"]["document"]
    need(expected["seconds"] == 10585 and len(set(expected["pci"])) == 8 and worker["epoch"] == 1795538478.6295807
         and worker["step"] == "1103606.2874" and worker["node"] == NODE and worker["synthetic"] is True
         and set(worker["pci"]) == set(expected["pci"]), "T07 preserved worker identity differs")
    need(gpu[0] == "1103606.2874" and gpu[1] == "training80-dcu-standby-3fb996b1b261-f13r2n07"
         and gpu[5:7] == [NODE, "cpu=8,gres/dcu=8,mem=16G,node=1"], "T07 predecessor accounting differs")
    if preparation_only:
        need(gpu[2:4] == ["RUNNING","0:0"] and target["terminal"] is None
             and set(target["live"]) == set(target["squeue"]) == {gpu[0]},
             "T07 preparation is only for the still-running predecessor")
    else:
        need(gpu[2:4] == ["COMPLETED","0:0"] and gpu[8] != "Unknown" and target["terminal"] is not None
             and target["live"] == target["squeue"] == {}, "T07 predecessor has not naturally released the node")
        term = target["terminal"]["document"]
        need(term["state"] == "WORKER_RETURNED" and term["step"] == gpu[0] and term["node"] == NODE
             and worker["epoch"] + expected["seconds"] <= term["epoch"] <= obs["epoch"] + 2,
             "T07 same worker has no natural return")
    holds = mod.read_queue_holds(REPO/"local-only/tuning/queue.sqlite3")
    jobs = json.loads((REPO/"coordination/training/training80-backlog-v1.json").read_bytes())["items"]
    known = set(mod.allocation(mod.TRAINING80_LEDGER.read_bytes())[2])
    known.update(n for job in jobs for n in job["placement"])
    original, _, _ = mod.read_original_holds(REPO, jobs, known, holds)
    need(not any(NODE in row["nodes"] for row in (*holds, *original.values()))
         and not any(j["state"] == "ready" and NODE in j["placement"] for j in jobs),
         "active or ready project work has priority over T07 standby")
    return gpu[0], target["terminal"]["SHA256"] if not preparation_only else None


def main():
    need(sys.argv[1:] == ["--capture-current-and-dispatch"], "explicit once-only scope is required")
    need(not MARKER.exists(), "T07 already has a reservation; reconcile its original receipt")
    need(digest(READER.read_bytes()) == READER_PIN, "qualified predecessor observation source differs")
    started = time.monotonic()
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(READER)], cwd=REPO,
                            capture_output=True, timeout=33)
    received = time.monotonic()
    need(result.returncode == 0 and 0 < len(result.stdout) <= 8192
         and 0 <= received-started <= 28, "current predecessor observation failed")
    found = json.loads(result.stdout)
    proof = (PRIVATE / found["file"]).resolve(strict=True)
    pin = found["SHA256"]
    old_gpu, terminal_pin = prior_proof(proof, pin)
    recorded = time.monotonic()
    need(0 <= recorded-received <= 3, "local predecessor observation aged before reservation")
    save(MARKER, json_data(dict(predecessor=old_gpu, old_request=PREVIOUS, node=NODE,
        received_monotonic=received, roundtrip_s=received-started, created_monotonic=recorded,
        proof_SHA256=pin, prior_worker_terminal_SHA256=terminal_pin, route="qualified-direct-refill",
        runner_SHA256=digest(RUNNER.read_bytes()), replay_allowed=False)))
    argv = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--fresh-runtime"]
    env = dict(os.environ, HYGON_STANDBY_ONLY_NODES=NODE, HYGON_STANDBY_VIA_COMPUTE="0")
    try:
        call = subprocess.run(argv, cwd=WORK, env=env, capture_output=True, timeout=248)
    except subprocess.TimeoutExpired as err:
        call = SimpleNamespace(returncode=None, stdout=err.stdout or b"", stderr=err.stderr or b"")
    observed=[]
    for line in call.stdout.splitlines():
        if 0 < len(line) <= 65536 and line.startswith(b"{"):
            try:
                data=json.loads(line)
            except ValueError:
                continue
            if isinstance(data,dict):
                observed.append(data)
    out = dict(rc=call.returncode,stdout_SHA256=digest(call.stdout),stderr_SHA256=digest(call.stderr),
               observations=observed,admission_transport_failure=any(b"Bounded admission did not complete" in l for l in call.stdout.splitlines()),
               replay_allowed=False)
    p = PRIVATE / "training80-T07-predecessor2874-direct-guarded-once-result.json"
    pin = save(p,json_data(out))
    compact=[]
    for row in observed:
        if "admission" in row:
            a=row["admission"]
            compact.append(dict(device_request=a.get("request"),fresh=a.get("fresh"),
                                target=a.get("exceptions",{}).get(NODE),admitted_cards=a.get("clear_cards")))
        else:
            compact.append({k:v for k,v in row.items() if k in ("standby_no_dispatch","request","remote_outcome","reason",
                 "submit_records","launch_failures","card_steps_observed","planned_DCU","actual_card_steps","standby")})
    print(json.dumps(dict(file=p.name,SHA256=pin,rc=call.returncode,observations=compact,
         admission_transport_failure=out["admission_transport_failure"],stderr_SHA256=out["stderr_SHA256"])))
    return int(call.returncode != 0)


def fresh_runtime():
    need(os.environ.get("HYGON_STANDBY_ONLY_NODES") == NODE
         and os.environ.get("HYGON_STANDBY_VIA_COMPUTE") == "0", "T07 child placement differs")
    original = json.loads(MARKER.read_bytes())
    need(original["old_request"] == PREVIOUS and original["node"] == NODE and original["replay_allowed"] is False
         and 0 <= time.monotonic() - original["created_monotonic"] <= 10, "T07 local reservation differs")
    save(PRIVATE / "training80-T07-predecessor2874-fresh-runtime-token.json",
         json_data(dict(original_marker_SHA256=digest(MARKER.read_bytes()), node=NODE, replay_allowed=False)))
    path = REPO / "tools/run_standby_renewal_with_pinned_publication.py"
    need(digest(path.read_bytes()) == "e900334ac5be3a74fbde60f5c99457706dd9e802b7f749f0c375a1ecca48424e",
         "published early Git check differs")
    route = runpy.run_path(str(path))
    try:
        code, state = route["render"](), route["precheck"]()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"standby_no_dispatch": "publication precheck before device admission: " + type(error).__name__}))
        return
    os.chdir(WORK)
    exec(code, dict(__name__="__main__", certified_tip=state["head"],
                    pinned_members=route["ALL"],check_preverified_publication=lambda repo,tip,paths:
                        route["verify_bound"](state,repo,tip,paths)))


if __name__ == "__main__":
    try:
        if len(sys.argv) == 4 and sys.argv[1] == "--check-preparation":
            predecessor, pin = prior_proof(Path(sys.argv[2]).resolve(strict=True),sys.argv[3],preparation_only=True)
            print(json.dumps(dict(stage="only-running-predecessor-checked", predecessor=predecessor,
                                  worker_terminal_SHA256=pin, dispatched=False)))
        elif sys.argv[1:] == ["--fresh-runtime"]:
            fresh_runtime()
        else:
            raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, IndexError) as err:
        print(json.dumps(dict(kind=type(err).__name__, why_SHA256=digest(str(err).encode()))))
        raise SystemExit(1)
