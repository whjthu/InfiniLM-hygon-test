"""Audit the original GLM Socket/SHCA full-model screen after Slurm terminal evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

RUN = "6dae70744fdc40e3ba0ade61a5b1143c"
REQUEST = "52836d317bb14904bbbe77f546975bd6"
COMMIT = "009e1bc3748e5fe26a59b22b6cdd64ff603d978b"
CLAIM = "8518c9e6a726737ca2995b0d15d9477ec01f58a81d7374cd591eba07dccecb95"
LABEL = "GLM474-pair-" + RUN
ARMS = ("socket", "shca")
PARENT = "1103606"
READER_SHA = "0bb4f157c258b03af6d086300f619e914bd94d07809d5df20d5787c93f788e85"
TRANSPORT_SHA = "1f9ceec1199adfb04fb0cbc4f25c31e670960840bff316659434779a11f50741"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_verified(path, digest, substitutions=()):
    raw = path.read_bytes()
    require(sha(raw) == digest, "reference source pin differs")
    text = raw.decode()
    for old, new, count in substitutions:
        require(text.count(old) == count, "reference adapter boundary differs")
        text = text.replace(old, new)
    env = {"__name__": "pinned_reference"}
    exec(compile(text, str(path), "exec"), env)
    return env


def paired_reader(path):
    env = load_verified(path, READER_SHA, [
        ("3f322106a0530b6123efa76072b167a21eca1b8d95b68fa1c6654dfc091e5198", CLAIM, 1),
        ('LABEL + "-baseline-gate"', 'LABEL + "-socket-gate"', 1),
        ('verified["baseline"], verified["single-drain"]', 'verified["socket"], verified["shca"]', 1),
        ('{"43843": True, "43844": True}', '{"43843": True, "43844": True, "43861": True}', 1),
    ])
    env.update(RUN=RUN, LABEL=LABEL, ARMS=ARMS)
    return env


def launcher(root, read):
    dispatch = root / (LABEL + "-dispatch")
    request, request_sha = read(dispatch / "request.json")
    owner, owner_sha = read(dispatch / "launcher-owner.json")
    for row in (request, owner):
        require(all(row[k] == v for k, v in dict(run_id=RUN, request_id=REQUEST, claim_commit=COMMIT).items()),
                "foreign launch")
    require(request["claim_SHA256"] == CLAIM and request["epoch"] == owner["epoch"]
            and request["deadline"] - request["epoch"] == 117600, "wrong launch budget")
    terminal, terminal_sha = read(dispatch / "launcher-terminal.json")
    require(terminal["run_id"] == RUN and type(terminal["rc"]) is int
            and terminal["finished"] >= owner["epoch"], "wrong launcher terminal")
    return dict(terminal, terminal_SHA256=terminal_sha, request_SHA256=request_sha, owner_SHA256=owner_sha)


def verify_transport(root, arm, summary, progress, transport):
    folder = root / (LABEL + "-" + arm)
    gate = summary["accounting"]["gate"]
    network = transport["NETWORK"][arm]
    expected = {"infiniccl": transport["INFINICCL_SHA256"], "rccl": transport["RCCL_SHA256"]}
    if arm == "shca":
        expected["plugin"] = transport["PLUGIN_SHA256"]
    receipt = progress["transport"]
    ids = receipt["UID_SHA256"]
    require(progress["step"] == gate[0] and receipt["transport"] == arm
            and len(ids) == 2 and ids[0] != ids[1] and all(re.fullmatch("[0-9a-f]{64}", v) for v in ids),
            "wrong transport receipt or UID pair")
    providers = receipt["providers"]
    require({name: row["SHA256"] for name, row in providers.items()} == expected, "wrong provider pins")
    n = receipt["channel_log_prefix_bytes_per_rank"]
    require(n == 1024**2, "log prefix limit changed")
    chunks, log_pins = [], []
    for rank in range(16):
        raw = (folder / f"owner-{rank:02d}.json").read_bytes()
        require(sha(raw) == summary["owners"][str(rank)]["SHA256"], "owner changed during audit")
        owner = json.loads(raw)
        require(owner["passed"] and owner["completed"] and owner["transport"] == arm
                and owner["UID_SHA256"] == ids and owner["transport_providers"] == providers,
                "rank transport evidence differs")
        with (folder / f"owner-{rank:02d}.log").open("rb") as f:
            prefix = f.read(n)
        chunks.append(prefix.decode("utf-8", errors="replace"))
        log_pins.append(sha(prefix))
    found = transport["network_channels"]("\n".join(chunks), arm)
    require(found == receipt["network"] and found["network"] == network, "transport channel report differs")
    return dict(network=found, providers=providers, log_prefix_SHA256=log_pins, UID_SHA256=ids)


def audit(root, reader, transport, rows):
    run = launcher(root, reader["read"])
    indexed = reader["accounting"](rows)
    socket_gate = indexed.get(LABEL + "-socket-gate")
    require(socket_gate and socket_gate[0] == "1103606.102", "wrong original Socket handle")
    require(all(reader["is_terminal"](row) for row in indexed.values()), "original steps not terminal")
    result = reader["summarize"](root, rows)
    progress_path = root / (LABEL + "-progress.json")
    progress, progress_sha = reader["read"](progress_path) if progress_path.is_file() else ({}, None)
    require(set(progress) <= set(ARMS), "unknown launcher progress arm")
    for arm, item in result["arms"].items():
        item["planned_screen_requests"] = 32
        item["verified_screen_requests"] = sum(r["completed_verified"] for r in item.get("screen", {}).values())
        item["all_rank_normal_close"] = len(item.get("owners", {})) == 16 and all(
            r["identity_verified"] and r["normal_close"] for r in item["owners"].values())
        item["transport_verified"] = arm in progress and bool(item["all_rank_normal_close"])
        if item["transport_verified"]:
            item["transport"] = verify_transport(root, arm, item, progress[arm], transport)
    ordered = [indexed.get(f"{LABEL}-{arm}-{phase}") for arm in ARMS for phase in ("pre", "gate", "post")]
    numbers = [int(row[0].rsplit(".", 1)[1]) for row in ordered if row]
    complete = (run["rc"] == 0 and len(numbers) == 6 and numbers == sorted(set(numbers))
                and all(row[2:4] == ["COMPLETED", "0:0"] for row in ordered)
                and all(v["post_verified"] and v["all_rank_normal_close"] and v["transport_verified"]
                        and v["verified_screen_requests"] == 32 and v["diagnostic_completed"] == 1
                        and all(v.get(phase + "_receipt", {}).get(key) is True
                                for phase in ("gate", "post") for key in ("accounting_matches", "terminal", "passed"))
                        for v in result["arms"].values()))
    result.update(launcher=run, progress_SHA256=progress_sha, complete_engineering_pair=complete,
                  formal_quality_qualified=False, service_qualified=False, fleet_busy_qualified=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("original_reader", type=Path)
    parser.add_argument("transport_reader", type=Path)
    args = parser.parse_args()
    reader = paired_reader(args.original_reader)
    transport = load_verified(args.transport_reader, TRANSPORT_SHA)
    raw = subprocess.check_output(
        ["sacct", "-n", "-P", "-j", PARENT, "--format=" + reader["COLUMNS"]], text=True, timeout=12)
    rows = [line.split("|") for line in raw.splitlines()]
    print(json.dumps(audit(args.results_root, reader, transport, rows), allow_nan=False))


if __name__ == "__main__":
    main()
