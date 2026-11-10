"""Recompute the original C4 comparison from the owned run and Slurm cleanup."""

import json

from alias_policy import SCHEDULE
from alias_summary import summarize
from diagnostic_binding import LIBRARY, NATIVE
from pilot_inputs import compact, require, resolve, sha
from pilot_observation import evaluate, wire_identity_and_first_text
from split_admission import validate_before

RUN = "i03-deep-c4-alias-http-v1"
STEP = "1087656.72"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def load(path, cap=16 * 1024**2):
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(len(raw) <= cap, "private artifact exceeds reader bound")
    return raw


def consume(folder, before, leased, post, terminal, root, shared, spec, intent, config_sha,
            *, decode_sse, grade):
    require(terminal["step"] == leased["step"] == STEP and terminal["label"] == RUN
            and terminal["state"].split()[0] in TERMINAL, "unconfirmed original terminal step")
    require(leased["holder_scan_location"] == "slurm_host_before_container"
            and leased["before_sha256"] == sha(json.dumps(before, sort_keys=True, separators=(",", ":")).encode()),
            "wrong device lease")
    validate_before(before, leased)
    require(post["kind"] == "i03_device_snapshot" and post["mode"] == "cpu"
            and post["node"] == leased["node"] and post["boot_id"] == leased["boot_id"]
            and post["pci"] == leased["pci"] and post["kfd_holder_count"] == 0
            and post["step"].startswith("1087656.")
            and int(post["step"].split(".")[1]) > int(STEP.split(".")[1])
            and post["monotonic_ns"] > leased["monotonic_ns"]
            and post["accounting_state"] == "COMPLETED" and post["exit_code"] == "0:0",
            "wrong terminal GPU cleanup")
    result = dict(step=STEP, terminal=terminal, before_step=before["step"], post_step=post["step"],
                  independent_post_clean=True, planned=len(SCHEDULE), normal_service_completion=False)
    artifacts = {}

    def read(name):
        raw = load(folder / name)
        artifacts[name] = sha(raw)
        return raw

    if (folder / "progress.json").is_file():
        progress = json.loads(read("progress.json"))
        require(len(progress) == len(SCHEDULE), "wrong progress slot inventory")
        result["slots"] = [slot["state"] for slot in progress]
    if not (folder / "alias-summary.json").is_file() or terminal["state"] != "COMPLETED" or terminal["exit_code"] != "0:0":
        result["artifacts_SHA256"] = artifacts
        return result

    require(result.get("slots") == ["RECORDED"] * len(SCHEDULE), "incomplete terminal requests")
    require(sha(read("service-config.json")) == config_sha, "foreign runtime configuration")
    owner, ready, engine, closed = [json.loads(read(f"{RUN}/{name}")) for name in
                                    ("owner.json", "MODEL_READY.json", "ready-engine.json", "closed-engine.json")]
    identity = owner["identity"]
    require(ready["identity"] == engine["identity"] == closed["identity"] == identity
            and identity["actual_step"] == STEP and identity["run"] == RUN
            and identity["provider_config_SHA256"] == config_sha
            and identity["source_manifest_SHA256"] == "bc7541b86afd15894d374c774ca3ed5093c623134bc493467518e4b7b3a2a98a"
            and identity["gpu_start_epoch"] == intent["epoch"]
            and engine["loaded_key_count"] == 67569 and engine["capacity"] == 32768 and engine["rows"] == 1
            and ready["Model_READY"] is True and ready["Native_internal_owner_count"] == 8
            and owner["Native_SHA256"] == ready["Native_SHA256"] == NATIVE
            and set(owner["actual_providers_before_weights"].values()) == {str(LIBRARY)}
            and owner["completed"] is True and owner["Model_READY"] is False
            and closed["normal"] is True and closed["engine_destroy_returned_zero"] is True
            and closed["observed_all8_threads_exited"] is True and len(owner["requests"]) == len(SCHEDULE),
            "bad engine identity or teardown")
    inputs = resolve(root, spec, host_shared=shared)
    rows = []
    for index, (case_index, _) in enumerate(SCHEDULE):
        entry, _, case = inputs[case_index]
        wire = read(f"request-{index}.wire")
        reads = [json.loads(line) for line in read(f"request-{index}.clock.jsonl").splitlines()]
        request_id, _ = wire_identity_and_first_text(wire, reads)
        row = owner["requests"][index]
        require(row["request_id"] == request_id and row["r454_index"] == index,
                "wrong original request order")
        stats = evaluate(entry, case, wire, reads, row, decode_sse=decode_sse, grade=grade)
        require(compact(stats) == compact(json.loads(read(f"request-{index}.summary.json"))),
                "request metric recomputation differs")
        rows.append(stats)
    summary = summarize(owner["requests"], rows)
    require(compact(summary) == compact(json.loads(read("alias-summary.json"))),
            "paired metric recomputation differs")
    for stats, row in zip(rows, owner["requests"]):
        stats.pop("request_id")
        stats.pop("backend_ITL_s")
        stats["per_owner_C4_counters"] = row["history_alias"]["counters"]
    result.update(normal_service_completion=True, requests=rows, comparison=summary,
                  artifacts_SHA256=artifacts)
    return result
