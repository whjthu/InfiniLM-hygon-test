"""Independently verify the original ordinal 150 Slurm pair and print bounded evidence."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

RUN = "62e402dd71fa418ea81f43684863a53c"
REQUEST = "ee478d50c5d540d5a1a7f6b903722384"
CLAIM_COMMIT = "eea2b60cd23eb67d9d8c52d2910b3da8c41b9cf2"
LABEL = "GLM474-pair-" + RUN
ADMISSION_LABEL = "GLM474-ordinal150-" + RUN + "-admission"
STAGE_LABEL = "glm474-transport-ordinal150-full-model-v1-" + REQUEST
PARENT = "1103606"
NODES = ("f13r1n05", "f13r1n07")
ARMS = ("socket", "shca")
COLUMNS = "JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160"
TERMINAL_STATES = frozenset(("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED",
                             "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "REVOKED"))


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def bounded_json(path, max_size=8 << 20):
    with path.open("rb") as stream:
        raw = stream.read(max_size + 1)
    need(0 < len(raw) <= max_size, "original evidence exceeds expected bound")
    return hash_bytes(raw), json.loads(raw)


def slurm(*args):
    output = subprocess.check_output(args, timeout=12, text=True)
    need(len(output) <= 1 << 20, "Slurm output exceeds expected scope")
    return output.splitlines()


def expected_steps():
    return {
        ADMISSION_LABEL: 0,
        **{f"{LABEL}-{arm}-{phase}": (16 if phase == "gate" else 0)
           for arm in ARMS for phase in ("pre", "gate", "post")},
    }


def classify(rows, expand):
    expected = expected_steps()
    fields = {}
    handles = set()
    for line in rows:
        row = line.split("|")
        if len(row) < 2 or row[1] not in expected:
            continue
        need(len(row) == 7 and row[1] not in fields and row[0] not in handles,
             "duplicate or malformed original Slurm identity")
        parent, dot, suffix = row[0].partition(".")
        need(parent == PARENT and dot and suffix.isdecimal(), "wrong original parent or step")
        state = row[2].split()[0].rstrip("+") if row[2] else ""
        terminal = state in TERMINAL_STATES
        cards = [x for x in row[6].split(",") if x.startswith("gres/dcu=")]
        want = expected[row[1]]
        if terminal and row[2:4] == ["COMPLETED", "0:0"]:
            need(tuple(sorted(expand(row[5]))) == tuple(sorted(NODES)), "wrong original node set")
            need(cards == ([f"gres/dcu={want}"] if want else []), "wrong original DCU count")
        fields[row[1]] = dict(row=row, terminal=terminal, success=terminal and row[2:4] == ["COMPLETED", "0:0"])
        handles.add(row[0])
    return fields


def validate_dispatch(stage, refs):
    request_digest, request = bounded_json(stage / "request.json")
    owner_digest, owner = bounded_json(stage / "launcher-owner.json")
    uid = dict(run_id=RUN, request_id=REQUEST, claim_commit=CLAIM_COMMIT)
    need(all(all(v[k] == value for k, value in uid.items()) for v in (request, owner)),
         "controller request identity differs")
    need(owner["epoch"] == request["epoch"] and owner["stage"] == "LAUNCHER_STARTED"
         and owner["PID"] > 0 and owner["startticks"] > 0, "controller ownership not recorded")
    need(hash_bytes((stage / "SOURCE_FILES.json").read_bytes()) == refs["manifest_SHA256"],
         "remote installed package manifest differs")
    for name, pin in refs["host_file_SHA256"].items():
        need(hash_bytes((stage / name).read_bytes()) == pin, "remote audit dependency differs")
    return {"request_SHA256": request_digest, "launcher_owner_SHA256": owner_digest}


def audit(result_root, refs, slurm_query=slurm):
    stage = result_root / STAGE_LABEL
    read = bounded_json(stage / "launcher-terminal.json") if (stage / "launcher-terminal.json").is_file() else None
    rows = slurm_query("sacct","-nP","-j",PARENT,"--format="+COLUMNS)
    accounting = classify(rows, lambda expression: slurm_query("scontrol","show","hostnames",expression))
    out = dict(state="WAITING_ORIGINAL", observed_UTC=datetime.now(timezone.utc).isoformat(),
               original_request=REQUEST, original_run=RUN, accounting=accounting,
               formal_inference_pass=False, promotion=False)
    if read is None or any(not row["terminal"] for row in accounting.values()):
        return out
    terminal_hash, terminal = read
    need(terminal["run_id"] == RUN and type(terminal["rc"]) is int, "invalid launcher terminal")
    out.update(validate_dispatch(stage, refs), launcher_terminal_SHA256=terminal_hash, launcher_rc=terminal["rc"])
    if terminal["rc"] != 0:
        possible_missing = []
        for arm in ARMS:
            folder = result_root / f"{LABEL}-{arm}"
            for phase in ("pre", "gate", "post"):
                if (folder / (phase+"-receipt.json")).is_file() and f"{LABEL}-{arm}-{phase}" not in accounting:
                    possible_missing.append(f"{arm}-{phase}")
        out.update(state="FAILED_ORIGINAL_LAUNCHER", accepted_pair=False,
                   any_active_or_unreconciled_possible_step=bool(possible_missing),
                   possible_unaccounted_steps=possible_missing,
                   required_canonical_post_unverified=[
                       arm for arm in ARMS if f"{LABEL}-{arm}-gate" in accounting
                       and not accounting.get(f"{LABEL}-{arm}-post", {}).get("success")])
        return out
    need(set(accounting) == set(expected_steps()) and all(row["success"] for row in accounting.values()),
         "successful controller lacks complete successful Slurm steps")

    admission = accounting[ADMISSION_LABEL]["row"]
    admission_hash, admission_record = bounded_json(stage / "cpu-admission-receipt.json")
    need(admission_record["passed"] is True and admission_record["terminal"] is True
         and admission_record["row"] == admission, "original zero-card admission differs")
    admission_pins = {}
    for rank, node in enumerate(NODES):
        digest, record = bounded_json(stage / f"{node}-container.json")
        need(record["node"] == node and record["rank"] == rank and record["step"] == admission[0]
             and record["passed"] is True and record["cards"] == 0
             and record["weight_tensor_payload_bytes_read"] == 0
             and record["Native_SHA256"] == refs["Native_SHA256"], "invalid original admission owner")
        admission_pins[node] = digest
    out.update(cpu_admission_SHA256=admission_hash, cpu_owner_admission_SHA256=admission_pins)
    sys.path.insert(0, str(stage))
    importlib.invalidate_caches()
    host = importlib.import_module("host_launch")
    reader = importlib.import_module("transport_result")
    need(Path(host.__file__).resolve().parent == stage.resolve()
         and Path(reader.__file__).resolve().parent == stage.resolve(), "audit module came from different run")
    owner_results = {}
    prior_gate_pins = None
    for arm in ARMS:
        folder = result_root / f"{LABEL}-{arm}"
        before = host.evidence(folder,"pre",accounting[f"{LABEL}-{arm}-pre"]["row"][0])
        step = accounting[f"{LABEL}-{arm}-gate"]["row"][0]
        gate = host.evidence(folder,"gate",step)
        post = host.evidence(folder,"post",accounting[f"{LABEL}-{arm}-post"]["row"][0])
        host.verify_released(before, gate)
        host.verify_released(gate, post)
        gate_pins = {node: gate[node]["pci_to_kfd"] for node in NODES}
        need(prior_gate_pins in (None, gate_pins), "paired device mapping changed")
        prior_gate_pins = gate_pins
        manifest_hash, config = bounded_json(folder/"RUN_MANIFEST.json")
        host.verify_manifest(config, RUN, arm, config["epoch"])
        run = host.verify_owners(folder,arm,manifest_hash,step,gate)
        need(run["network"]["network"] == ("Socket" if arm=="socket" else "IBext_v8"),
             "cross-node route was not verified")
        owner_results[arm]=run["sample43"]
    independent = reader.compare(owner_results)
    controller_hash, host_summary = bounded_json(stage/"paired-summary.json")
    need(all(host_summary[key] == independent[key] for key in independent),
         "independent request result differs from controller summary")
    out.update(state="FULL_MODEL_ORDINAL150_PAIR_VERIFIED", accepted_pair=True, independent=independent,
               controller_pair_SHA256=controller_hash, complete_requests=2, normal_closes=32,
               both_arms_clean_post=True)
    return out


def main():
    root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
    pins = json.loads(Path(__file__).with_name("BINDING.json").read_bytes())
    print(json.dumps(audit(root,pins),allow_nan=False))


if __name__ == "__main__":
    main()
