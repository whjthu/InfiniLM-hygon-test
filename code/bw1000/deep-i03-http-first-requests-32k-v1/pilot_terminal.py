"""Independently consume one terminal public pilot without returning private streams or IDs."""

import json
import math

from pilot_inputs import compact, require, resolve, sha
from pilot_observation import evaluate, wire_identity_and_first_text

RUN = "i03-deep-r454-http-first2-v1"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def load(path, cap=16 * 1024**2):
    with path.open("rb") as f:
        raw = f.read(cap + 1)
    require(len(raw) <= cap, "private artifact exceeds size bound")
    return raw


def one(rows, predicate, optional=False):
    found = [row for row in rows if predicate(row)]
    require(len(found) == 1 or optional and not found, "missing or ambiguous receipt row")
    return found[0] if found else None


def post_clean(before, leased, post, *, validate_before):
    validate_before(before, leased)
    return bool(post and post["kind"] == "i03_device_snapshot" and post["mode"] == "cpu"
                and post["node"] == leased["node"] and post["boot_id"] == leased["boot_id"]
                and post["pci"] == leased["pci"] and post["kfd_holder_count"] == 0
                and post["step"].startswith("1087656.")
                and int(post["step"].split(".")[1]) > int(leased["step"].split(".")[1])
                and post["monotonic_ns"] > leased["monotonic_ns"]
                and post["accounting_state"] == "COMPLETED" and post["exit_code"] == "0:0")


def digest_map(folder, pins):
    verified = {}
    for name, expected in pins.items():
        item = folder / name
        require(item.resolve().is_relative_to(folder.resolve()), "artifact outside exact private run")
        digest = sha(load(item))
        require(digest == expected, "private artifact digest differs")
        verified[name] = digest
    return verified


def latency_percentiles(values):
    order = sorted(values)
    return {f"p{p}": order[max(0, math.ceil(p * len(order) / 100) - 1)] if order else None
            for p in (50, 95, 99)}


def consume(folder, records, terminal, host_root, host_shared, spec, *, validate_before, decode_sse, grade):
    step = terminal["step"]
    require(terminal["state"].split()[0] in TERMINAL and step.startswith("1087656.")
            and terminal["label"] == RUN, "unverified original terminal accounting")
    before = one(records, lambda row: row.get("controller") == "before")["snapshot"]
    leased = one(records, lambda row: row.get("kind") == "i03_device_snapshot")
    require(leased["step"] == step and leased["holder_scan_location"] == "slurm_host_before_container"
            and leased["before_sha256"] == sha(json.dumps(before, sort_keys=True, separators=(",", ":")).encode()),
            "different original lease")
    post = one(records, lambda row: row.get("controller") == "post", optional=True)
    clean = post_clean(before, leased, post["snapshot"] if post else None, validate_before=validate_before)
    complete = one(records, lambda row: row.get("kind") == "i03_r454_http" and row.get("stage") == "complete",
                   optional=True)
    if not complete:
        states = ["MISSING", "MISSING"]
        path = folder / "progress.json"
        if path.is_file():
            progress = json.loads(load(path))
            require([row["case_id"] for row in progress] == [row["case_id"] for row in spec["selected"]],
                    "foreign progress file")
            states = [row["state"] for row in progress]
        return dict(step=step, terminal=terminal, planned=2, independent_post_clean=clean,
                    normal_service_completion=False, slot_states=states)
    require(terminal["state"] == "COMPLETED" and terminal["exit_code"] == "0:0"
            and complete["step"] == step and complete["planned"] == complete["recorded"] == 2,
            "missing completed service or original Slurm proof")
    expected = ["progress.json", "service-config.json", "service.log"] + [
        f"request-{i}{ext}" for i in range(2) for ext in (".wire", ".clock.jsonl", ".summary.json")]
    expected += [RUN + "/" + name for name in ("owner.json", "MODEL_READY.json", "ready-engine.json",
                                               "closed-engine.json")]
    require(set(expected) == set(complete["artifacts_SHA256"]), "original evidence inventory differs")
    hashes = digest_map(folder, complete["artifacts_SHA256"])
    engine = json.loads(load(folder / RUN / "ready-engine.json"))
    owner = json.loads(load(folder / RUN / "owner.json"))
    ready = json.loads(load(folder / RUN / "MODEL_READY.json"))
    closed = json.loads(load(folder / RUN / "closed-engine.json"))
    identity = owner["identity"]
    require(engine["identity"] == ready["identity"] == closed["identity"] == identity
            and identity["run"] == RUN and identity["actual_step"] == step
            and identity["provider_config_SHA256"] == hashes["service-config.json"]
            and engine["loaded_key_count"] == 67569 and engine["rows"] == 1 and engine["capacity"] == 32768
            and ready["Model_READY"] is True and ready["Native_internal_owner_count"] == 8
            and closed["normal"] is True and closed["engine_destroy_returned_zero"] is True
            and closed["observed_all8_threads_exited"] is True, "ready or normal destroy evidence differs")
    selected = resolve(host_root, spec, host_shared=host_shared)
    require(len(owner["requests"]) == len(selected) == 2, "actual request denominator differs")
    public = []
    for i, (entry, _, case) in enumerate(selected):
        wire = load(folder / f"request-{i}.wire")
        clocks = [json.loads(line) for line in load(folder / f"request-{i}.clock.jsonl").splitlines()]
        request_id, _ = wire_identity_and_first_text(wire, clocks)
        row = one(owner["requests"], lambda item: item["request_id"] == request_id)
        require(row["r454_index"] == i, "native request sequence differs")
        metric = evaluate(entry, case, wire, clocks, row, decode_sse=decode_sse, grade=grade)
        require(compact(metric) == compact(json.loads(load(folder / f"request-{i}.summary.json"))),
                "independent metric reconstruction differs")
        metric.pop("request_id")
        metric["backend_ITL_nearest_rank_s"] = latency_percentiles(metric.pop("backend_ITL_s"))
        public.append(metric)
    return dict(step=step, terminal=terminal, independent_post_clean=clean, planned=2,
                normal_service_completion=True, requests=public, artifacts_SHA256=hashes,
                measured_user_token_timestamps=False, broad_quality_qualified=False)
