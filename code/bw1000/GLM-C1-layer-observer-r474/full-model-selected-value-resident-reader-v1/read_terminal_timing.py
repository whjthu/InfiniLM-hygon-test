"""Read the original GLM .141 terminal on the cluster without loading a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
RUN = "684e399249074260a3edf9fcf96cd4e6"
LABEL = "GLM474-selected-resident-" + RUN
STEP = "1103606.141"
CLAIM = "8237a87c13bd7f46aabbc122e708ba9db18eb2795eeae8bd7be86138578fc1ec"
NATIVE = "bc1513a10a21860db41fab192093acb9850f4fb4412838c9940f6dfac130fa4e"
HOST_MANIFEST = "51b968e6e0382747981dfa60b5766cddfacc9be7eedd5af7f53ee20a78407bcd"
WORKER_MANIFEST = "69a8be851da5d29f5a93ed57933ae78ae71e11a9fe141c2405714a6e146fc7b6"
NODES = ("f13r1n05", "f13r1n07")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path, limit=32 * 1024**2):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, f"{path.name} exceeds bound")
    return raw, hashlib.sha256(raw).hexdigest()


def read_json(path, limit=32 * 1024**2):
    raw, digest = read(path, limit)
    return json.loads(raw), digest


def accounting():
    command = ["sacct", "-n", "-P", "-j", "1103606",
               "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList"]
    text = subprocess.check_output(command, text=True, timeout=12)
    labels = {LABEL + "-" + stage: stage for stage in ("pre", "gate", "post")}
    result = {}
    for line in text.splitlines():
        row = line.split("|")
        if len(row) != 6 or row[1] not in labels:
            continue
        stage = labels[row[1]]
        require(stage not in result and re.fullmatch(r"1103606\.\d+", row[0]),
                "ambiguous accounting row")
        result[stage] = row
    return result


def is_terminal(row):
    return bool(row and row[2].split() and row[2].split()[0] in TERMINAL)


def nearest_rank(values, percent):
    return sorted(values)[math.ceil(len(values) * percent / 100) - 1] if values else None


def visible_timing(ids, stamps, eos, special):
    require(len(ids) == len(stamps) and all(a < b for a, b in zip(stamps, stamps[1:])),
            "token events do not have strictly increasing timestamps")
    visible = [ns for token, ns in zip(ids, stamps) if token not in eos and token not in special]
    itl = [(b - a) / 1e6 for a, b in zip(visible, visible[1:])]
    return dict(visible_tokens=len(visible), intervals=len(itl),
                itl_p95_ms=nearest_rank(itl, 95), itl_p99_ms=nearest_rank(itl, 99),
                decode_tok_s=len(itl)*1e9/(visible[-1]-visible[0]) if itl else None)


def terminal_summary(root):
    rows = accounting()
    gpu = rows.get("gate")
    require(gpu and gpu[0] == STEP and is_terminal(gpu), "original GPU step is not terminal")
    out = root / LABEL
    result = dict(accounting=rows, step=STEP, planned_requests=8, planned_measured_pairs=3,
                  original_collector=None, cleanup_verified=False, timed_backend_requests=0,
                  online_SLO_qualified=False, broad_quality_qualified=False)
    try:
        outcome, outcome_sha = read_json(out / "outcome.json")
    except FileNotFoundError:
        return result
    require(outcome["run_id"] == RUN and outcome["claim_SHA256"] == CLAIM, "collector identity changed")
    result.update(original_collector=outcome["passed"], outcome_SHA256=outcome_sha)
    if not outcome["terminal"]:
        return result
    require(outcome["stages"]["gate"]["row"] == gpu, "collector GPU accounting differs")
    if outcome.get("cleanup_verified"):
        for stage in ("pre", "post"):
            row = rows.get(stage)
            require(row and row[2:4] == ["COMPLETED", "0:0"]
                    and outcome["stages"][stage]["row"] == row, f"{stage} accounting differs")
        for node in NODES:
            pre, _ = read_json(out / "pre" / (node + ".json"), 64 * 1024)
            post, _ = read_json(out / "post" / (node + ".json"), 64 * 1024)
            require(pre["step"] == rows["pre"][0] and post["step"] == rows["post"][0]
                    and pre["node"] == post["node"] == node and pre["boot_id"] == post["boot_id"]
                    and pre["pci"] == post["pci"] and len(set(pre["pci"])) == 8
                    and post["monotonic_ns"] >= pre["monotonic_ns"]
                    and post["kfd_holder_count"] == 0
                    and set(post["ports_bindable"]) == {"43843", "43844", "43861"}
                    and all(value is True for value in post["ports_bindable"].values()),
                    "independent post-run device or port evidence differs")
        result["cleanup_verified"] = True
    if not (outcome["passed"] and result["cleanup_verified"] and gpu[2:4] == ["COMPLETED", "0:0"]):
        return result

    package = root / (LABEL + "-dispatch") / "src"
    for manifest, pinned in (("HOST_MANIFEST.json", HOST_MANIFEST), ("SOURCE_MANIFEST.json", WORKER_MANIFEST)):
        data, digest = read_json(package / manifest, 64 * 1024)
        require(digest == pinned, "original source manifest changed")
        for name, expected in data["files_SHA256"].items():
            require(Path(name).name == name, "invalid source name")
            require(read(package / name, 1024**2)[1] == expected, f"{name} digest differs")
    sys.path.insert(0, str(package))
    from audit import verify_owners

    manifest, manifest_sha = read_json(out / "RUN_MANIFEST.json", 64 * 1024)
    require(manifest["run_id"] == RUN and manifest["claim_SHA256"] == CLAIM
            and manifest["Native_SHA256"] == NATIVE
            and manifest["source_manifest_SHA256"] == WORKER_MANIFEST, "run manifest differs")
    gate = {node: read_json(out / "gate" / (node + ".json"), 64 * 1024)[0] for node in NODES}
    for node, receipt in gate.items():
        mapping = receipt["pci_to_kfd"]
        require(receipt["node"] == node and receipt["step"] == STEP
                and len(set(mapping.values())) == len(mapping) == 8 and set(mapping) == set(receipt["pci"]),
                "leased equipment evidence differs")
    audit = verify_owners(out, manifest_sha, STEP, gate, NATIVE)
    require(audit == outcome["audit"], "independent audit differs from collector")
    binding, _ = read_json(package / "INPUT_BINDING.json", 1024**2)
    eos = set(binding["eos_token_ids"])
    special = set(binding["tokenizer"]["special_ids"])
    rank0, digest = read_json(out / "owner-00.json")
    require(digest == audit["rank_receipt_SHA256"]["owner-00.json"], "root receipt changed")
    timings = []
    for request, checked in zip(rank0["requests"], audit["root_requests"]):
        timing = visible_timing(request["generated_token_ids"], request["sampled_ns"], eos, special)
        require(timing["visible_tokens"] == checked["visible"]
                and timing["decode_tok_s"] == checked["decode_tok_s"], "visible decode differs from audit")
        timings.append(dict(ordinal=checked["ordinal"], arm=checked["arm"],
                            ttft_s=checked["ttft_s"], natural_EOS=checked["natural_EOS"],
                            cap_hit=checked["cap_hit"], score=checked["score"], backend_above_15=(timing["decode_tok_s"] > 15
                            if timing["decode_tok_s"] is not None else None), **timing))
    speeds = [row["decode_tok_s"] for row in timings[2:] if row["decode_tok_s"] is not None]
    result.update(timing_scope="Root sampled backend events, nearest-rank ITL percentiles, warmup excluded from summary",
                  timed_backend_requests=len(speeds), measured_backend_over_15=sum(speed > 15 for speed in speeds),
                  measured_backend_decode_median=statistics.median(speeds) if speeds else None,
                  timings=timings, packed_ranks_by_request=audit["packed_ranks_by_request"],
                  per_rank_counters=audit["per_rank_counters"], paired=audit["paired"], comparable_pairs=audit["comparable_pairs"],
                  median_ratio=audit["median_ratio"], singleton_prefill_exact=audit["singleton_prefill_exact"],
                  owner_receipts=audit["rank_receipt_SHA256"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(terminal_summary(ROOT), allow_nan=False))
