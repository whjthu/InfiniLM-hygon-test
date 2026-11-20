"""Independently verify and score the frozen Qwen USGS development run."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import subprocess


REPO = Path(__file__).resolve().parents[1]
LABEL = "training-qwen-usgs-dev80-base-adapter-v1"
PACKAGE = REPO / "code/bw1000" / LABEL
REQUEST = "bb61591025d94fdaba421ac15d5d4fa7"
PUBLICATION = "94812e070df7ed45d7141eed148d433b428a9782"
MANIFEST = "e2368f122d3ba4cb03b74ef1964cc417ee02eadaa3b16e0412386cda2bc491d4"
GOLD = "697956f4154f84911736744adecca73fd954aa6fb57d9e9e84b2c1592868d336"
ARMS = ("base", "domain")
PHASES = ("pre", "gpu", "post")
ACCOUNTING_COLUMNS = ["JobIDRaw", "JobName", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
IDS = [f"dev-{table:03d}-{question}" for table in range(1, 21) for question in range(1, 5)]
ORIGINAL = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results") / LABEL
TORCH_SHA256 = "9edf6b378be77655e0641130e18edfc32d4415d9355cf027bef3eba54aeeecc9"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def read(path, expected=None):
    need(path.is_file() and not path.is_symlink(), f"missing or linked evidence: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    need(len(raw) <= 1024 * 1024, f"evidence exceeds size limit: {path.name}")
    if expected is not None:
        need(digest(raw) == expected, f"evidence hash differs: {path.name}")
    return raw


def json_lines(raw):
    return [json.loads(line) for line in raw.splitlines()]


def snapshot(raw, node):
    result = json.loads(raw)
    need(result["node"] == node and type(result["KFD_holders"]) is int and result["KFD_holders"] == 0,
         "snapshot node or device ownership differs")
    devices = result["pci"]
    need(type(devices) is list and len(devices) == len(set(devices)) == 8 and
         all(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", x) for x in devices),
         "snapshot does not identify eight distinct PCI devices")
    need(re.fullmatch(r"[0-9a-f]{64}", result["boot_SHA256"]), "snapshot lacks a boot identity")
    return result


def check_accounting(row, claim, arm, phase):
    need(type(row) is list and len(row) == len(ACCOUNTING_COLUMNS), "invalid accounting columns")
    handle, label, state, exit_code, elapsed, node, resources = row
    need(re.fullmatch(re.escape(claim["parent"]) + r"\.[0-9]+", handle) and
         label == f"{LABEL}-{arm}-{phase}" and state == "COMPLETED" and exit_code == "0:0" and
         elapsed.isdecimal() and node == claim["placement"][arm], "accounting phase or placement differs")
    gres = [resource for resource in resources.split(",") if resource.startswith("gres/")]
    need(gres == (["gres/dcu=1"] if phase == "gpu" else []), "accounting device count differs")
    return handle


def read_arm(folder, claim, arm, result, live_rows, score, gold):
    need(result["arm"] == arm and result["passed"] is True, f"{arm} controller did not pass")
    nodes = claim["placement"]
    records, snapshots = {}, {}
    for phase in PHASES:
        stage = result[phase]
        need(stage["passed"] is True and type(stage["client_rc"]) is int and stage["client_rc"] == 0,
             f"{arm} {phase} client did not pass")
        handle = check_accounting(stage["row"], claim, arm, phase)
        need(live_rows.get(handle) == stage["row"], f"{arm} {phase} Slurm record differs")
        receipt = json.loads(read(folder / f"{arm}-{phase}-accounting.json"))
        need(receipt == stage, f"{arm} {phase} receipt differs")
        log = read(folder / f"{arm}-{phase}.log", stage["log_SHA256"])
        if phase != "gpu":
            snapshots[phase] = snapshot(log, nodes[arm])
        records[phase] = handle
    need(snapshots["pre"] == snapshots["post"], f"{arm} device or boot identity changed")

    output = folder / f"{arm}-outputs"
    run_raw = read(output / "generation-outcome.json")
    run = json.loads(run_raw)
    need(run["arm"] == arm and type(run["actual_DCU"]) is int and run["actual_DCU"] == 1 and
         run["precision"] == claim["dtype"] and run["attention"] == claim["attention"] and
         run["max_new_tokens"] == claim["max_tokens"] and run["planned_rows_per_arm"] == len(IDS) and
         run["greedy"] is True and run["batch_size"] == 1 and run["fresh_model_process"] is True,
         f"{arm} generation settings differ")
    provider = json.loads(read(output / "torch-provider.json"))
    need(provider == run["torch_provider"] and provider["SHA256"] == TORCH_SHA256,
         f"{arm} Torch receipt differs")
    signals = json_lines(read(output / "progress.jsonl"))
    need(signals and signals[0]["event"] == "plan" and signals[0]["arm"] == arm and
         signals[0]["planned_ids"] == IDS and signals[0]["planned_rows"] == len(IDS) and
         signals[0]["max_new_tokens"] == claim["max_tokens"], f"{arm} progress plan differs")
    generated = run["generated"]
    raw = read(output / f"{arm}-raw.jsonl", generated["raw_file_SHA256"])
    predictions = json_lines(raw)
    scoring_rows = json_lines(read(output / f"{arm}.jsonl", generated["scoring_file_SHA256"]))
    need(generated["rows"] == len(predictions) == len(IDS) and [r["id"] for r in predictions] == IDS,
         f"{arm} did not produce all planned rows in order")
    complete = [event for event in signals if event["event"] == "completed"]
    need(len(complete) == len(predictions) and all(
        event["arm"] == arm and event["id"] == row["id"] and event["finish_reason"] == row["finish_reason"]
        for event, row in zip(complete, predictions)), f"{arm} progress stream differs")
    counts = dict.fromkeys(("stop", "length", "empty"), 0)
    projected = []
    for row in predictions:
        reason = row["finish_reason"]
        answer = row["answer"]
        need(reason in ("stop", "length") and (answer is None or isinstance(answer, str)) and
             type(row["prompt_tokens"]) is int and row["prompt_tokens"] > 0 and
             type(row["completion_tokens"]) is int and 0 <= row["completion_tokens"] <= claim["max_tokens"],
             f"{arm} contains invalid native response evidence")
        counts[reason if answer or reason == "length" else "empty"] += 1
        projected.append({"id": row["id"], "answer": answer if reason == "stop" and answer else None})
    need(generated["outcome_counts"] == counts and scoring_rows == projected,
         f"{arm} raw prediction and score projection differ")
    return {
        "records": records,
        "prediction_SHA256": digest(raw),
        "generation_receipt_SHA256": digest(run_raw),
        "outcomes": score({row["id"]: row for row in predictions}, gold),
    }, provider


def paired(base, domain, ids):
    changes = Counter()
    for item in ids:
        a, b = base[item] == "correct", domain[item] == "correct"
        changes["both_correct" if a and b else "lost" if a else "gained" if b else "neither_correct"] += 1
    return {key: changes[key] for key in ("both_correct", "gained", "lost", "neither_correct")}


def verify(folder, dev_inputs, gold_path):
    manifest = json.loads(read(PACKAGE / "SOURCE_FILES.json", MANIFEST))["files"]
    for filename, pin in manifest.items():
        raw = read(folder / filename, pin["SHA256"])
        need(len(raw) == pin["bytes"], "installed packet size differs")
    claim = json.loads(read(folder / "CLAIM.json"))
    intent = json.loads(read(folder / "intent.json"))
    launch = json.loads(read(folder / "launch.json"))
    need(claim["request"] == intent["request"] == launch["request"] == REQUEST and
         intent["publication"] == launch["publication"] == PUBLICATION and
         launch == dict(intent, controller_pid=launch["controller_pid"]) and
         type(launch["controller_pid"]) is int and launch["controller_pid"] > 0 and
         intent["deadline"] - intent["epoch"] == claim["max_epoch_s"], "original launch identity differs")
    inventory = json.loads(read(folder / "DEV_INPUTS.json"))
    inputs = json_lines(read(dev_inputs, inventory["dataset_SHA256"]))
    gold_raw = read(gold_path, GOLD)
    gold = json_lines(gold_raw)
    need([row["id"] for row in inputs] == [row["id"] for row in gold] == IDS, "frozen development IDs differ")
    for table in range(1, 21):
        group = inputs[(table - 1) * 4:table * 4]
        need(all(len(r["messages"]) == 1 and r["messages"][0]["role"] == "user" and
                 len(r["images"]) == 1 and Path(r["images"][0]).name == f"table-{table:03d}.png" for r in group),
             "development table grouping or label isolation differs")
    outcome = json.loads(read(folder / "outcome.json"))
    need(outcome["request"] == REQUEST and set(outcome["results"]) == set(ARMS), "controller outcome identity differs")
    observation = json.loads(read(folder / "sacct-snapshot.json"))
    need(observation["columns"] == ACCOUNTING_COLUMNS and len(observation["rows"]) == len(ARMS) * len(PHASES),
         "independent Slurm snapshot is incomplete")
    live_rows = {row[0]: row for row in observation["rows"]}
    need(len(live_rows) == len(observation["rows"]), "independent Slurm snapshot has duplicate step handles")
    scorer = PACKAGE / "score_local.py"
    read(scorer, manifest[scorer.name]["SHA256"])
    score = runpy.run_path(str(scorer))["score"]
    gold_map = {row["id"]: row for row in gold}
    runs, providers = {}, []
    for arm in ARMS:
        runs[arm], provider = read_arm(folder, claim, arm, outcome["results"][arm], live_rows, score, gold_map)
        providers.append(provider)
    need(providers[0] == providers[1], "paired arms loaded different Torch providers")
    need(len({handle for run in runs.values() for handle in run["records"].values()}) == len(live_rows),
         "paired arms reused a Slurm step handle")
    a, b = (runs[arm]["outcomes"] for arm in ARMS)
    changes = paired(a, b, IDS)
    return dict(
        request=REQUEST,
        planned_questions=len(IDS), planned_tables=20, checkpoint_2=claim["checkpoint_2"],
        development_only=True, gold_SHA256=digest(gold_raw), dev_input_SHA256=inventory["dataset_SHA256"],
        arms={arm: {**run, "counts": dict(Counter(run["outcomes"].values()))} for arm, run in runs.items()},
        paired=changes, accuracy_difference=(changes["gained"]-changes["lost"])/len(IDS),
        per_table={f"{table:03d}": paired(a, b, IDS[(table-1)*4:table*4]) for table in range(1, 21)},
    )


def capture(destination):
    """Read the completed original on the cluster and copy its bounded evidence."""
    need(not destination.resolve().is_relative_to(ORIGINAL.resolve()),
         "capture output must be outside the original run")
    request = json.loads(read(ORIGINAL / "intent.json"))
    need(request["request"] == REQUEST and request["publication"] == PUBLICATION, "unexpected original intent")
    outcome = read(ORIGINAL / "outcome.json")
    result = json.loads(outcome)
    need(result["request"] == REQUEST and set(result["results"]) == set(ARMS) and
         all(stage["passed"] is True for stage in result["results"].values()), "original run did not complete")
    claim = json.loads(read(ORIGINAL / "CLAIM.json"))
    pinned = json.loads(read(ORIGINAL / "SOURCE_FILES.json", MANIFEST))["files"]
    files = {name: read(ORIGINAL / name, pin["SHA256"]) for name, pin in pinned.items()}
    files.update({name: read(ORIGINAL / name) for name in ("intent.json", "launch.json")})
    files["outcome.json"] = outcome
    expected = {}
    for arm in ARMS:
        for phase in PHASES:
            record = result["results"][arm][phase]
            handle = check_accounting(record["row"], claim, arm, phase)
            need(handle not in expected, "reused Slurm handle")
            expected[handle] = record["row"]
            filename = f"{arm}-{phase}"
            files[filename + "-accounting.json"] = read(ORIGINAL / (filename + "-accounting.json"))
            files[filename + ".log"] = read(ORIGINAL / (filename + ".log"), record["log_SHA256"])
        for name in ("generation-outcome.json", "torch-provider.json", "progress.jsonl",
                     f"{arm}-raw.jsonl", f"{arm}.jsonl"):
            key = f"{arm}-outputs/{name}"
            files[key] = read(ORIGINAL / key)
    fmt = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
    output = subprocess.check_output(
        ["sacct", "-nP", "-j", ",".join(expected), "--format=" + fmt], timeout=12, text=True
    )
    observed = [line.split("|") for line in output.splitlines() if line.split("|", 1)[0] in expected]
    need(len(observed) == len(expected) and {row[0]: row for row in observed} == expected,
         "original terminal accounting identity differs")
    destination.mkdir(mode=0o700)
    files["sacct-snapshot.json"] = json.dumps(dict(columns=ACCOUNTING_COLUMNS, rows=observed)).encode()
    for name, raw in files.items():
        path = destination / name
        path.parent.mkdir(mode=0o700, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    return {"request": REQUEST, "captured_files": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    collector = commands.add_parser("capture")
    collector.add_argument("--output", type=Path, required=True)
    verifier = commands.add_parser("verify")
    verifier.add_argument("--bundle", type=Path, required=True)
    verifier.add_argument("--dev-inputs", type=Path, required=True)
    verifier.add_argument("--gold", type=Path, required=True)
    options = parser.parse_args()
    report = (capture(options.output) if options.mode == "capture"
              else verify(options.bundle, options.dev_inputs, options.gold))
    print(json.dumps(report, indent=2, allow_nan=False))
