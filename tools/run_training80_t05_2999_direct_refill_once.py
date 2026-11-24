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
NODE = "f13r1n19"
RUNNER = WORK / "run-dcu-standby-cycle-20261123.ps1"
PREVIOUS = "7107a655388343c4b80639ab640736d0"
MARKER = PRIVATE / "training80-T05-predecessor2999-direct-guarded-once.json"
READER = PRIVATE / "read-training80-2997-2999-3001-exact-predecessors.py"
READER_PIN = "4e30b5df6771e42b4b7001d709c6dbd769986e57615390d69475cea4c34af7f8"


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
    need(proof.parent == PRIVATE and proof.name.startswith("training80-specialist-2999-exact-current-"),
         "not the T05 fresh terminal evidence")
    raw = proof.read_bytes()
    need(digest(raw) == pin and digest(RUNNER.read_bytes()) == "04f2bef937f98b35a29c0bfce7c199e56e049e001a0a5b92ecbf12878a358c73",
         "proof or qualified direct runner bytes differ")
    proof_record = json.loads(raw)
    obs = proof_record["observed"]
    target = obs["scopes"][NODE]
    need(proof_record["rc"] == 0 and obs["owner"] and obs["alloc_320"] and obs["parent_state"] == "RUNNING"
         and target["request"] == PREVIOUS and target["previous"] == "1103606.2999"
         and target["expected"]["SHA256"] == "e03ffc5719e9cc74b55bfa5659781997bb5cd6ee6d15ffa335fde00da909587b"
         and target["worker"]["SHA256"] == "2ebdf11ee6b1984bbb789a0080ebcb5a5f6b99f3a7b9f125196da9ad96763bcd",
         "T05 parent or original worker evidence differs")
    previous_pre = target["prior_probe_accounting"]
    need(previous_pre[:4] == ["1103606.2998", "training80-admit-b1881bf120bcee702a25900e3eb787de", "COMPLETED", "0:0"]
         and previous_pre[5:7] == [NODE, "cpu=1,mem=256M,node=1"], "prior T05 zero-GRES probe differs")
    gpu = target["accounting"]
    need(gpu[0] == "1103606.2999" and gpu[1] == "training80-dcu-standby-7107a6553883-f13r1n19"
         and gpu[5:7] == [NODE, "cpu=8,gres/dcu=8,mem=16G,node=1"], "T05 predecessor accounting differs")
    if preparation_only:
        need(gpu[2:4] == ["RUNNING","0:0"] and target["terminal"] is None
             and target["live"].get(gpu[0], {}).get("State") == "RUNNING",
             "T05 preparation is only for the still-running predecessor")
    else:
        need(gpu[2:4] == ["COMPLETED","0:0"] and gpu[8] != "Unknown" and target["terminal"] is not None
             and target["live"] == target["squeue"] == {}, "T05 predecessor has not naturally released the node")
        term = target["terminal"]["document"]
        need(term["state"] == "WORKER_RETURNED" and term["step"] == gpu[0] and term["node"] == NODE
             and 1795561492.429796 <= term["epoch"] <= obs["epoch"] + 2,
             "T05 same worker has no natural return")
    holds = mod.read_queue_holds(REPO/"local-only/tuning/queue.sqlite3")
    jobs = json.loads((REPO/"coordination/training/training80-backlog-v1.json").read_bytes())["items"]
    known = set(mod.allocation(mod.TRAINING80_LEDGER.read_bytes())[2])
    known.update(n for job in jobs for n in job["placement"])
    original, _, _ = mod.read_original_holds(REPO, jobs, known, holds)
    need(not any(NODE in row["nodes"] for row in (*holds, *original.values()))
         and not any(j["state"] == "ready" and NODE in j["placement"] for j in jobs),
         "active or ready project work has priority over T05 standby")
    return gpu[0], target["terminal"]["SHA256"] if not preparation_only else None


def main():
    need(sys.argv[1:] == ["--capture-current-and-dispatch"], "explicit once-only scope is required")
    need(not MARKER.exists(), "T05 already has a reservation; reconcile its original receipt")
    need(digest(READER.read_bytes()) == READER_PIN, "qualified predecessor observation source differs")
    started = time.monotonic()
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(READER), "--capture", "t05"], cwd=REPO,
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
    p = PRIVATE / "training80-T05-predecessor2999-direct-guarded-once-result.json"
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
         and os.environ.get("HYGON_STANDBY_VIA_COMPUTE") == "0", "T05 child placement differs")
    original = json.loads(MARKER.read_bytes())
    need(original["old_request"] == PREVIOUS and original["node"] == NODE and original["replay_allowed"] is False
         and 0 <= time.monotonic() - original["created_monotonic"] <= 10, "T05 local reservation differs")
    save(PRIVATE / "training80-T05-predecessor2999-fresh-runtime-token.json",
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
