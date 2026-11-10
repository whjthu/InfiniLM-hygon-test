"""Independently reconstruct the sparse HTTP pilot and verify original Slurm cleanup."""

import json

from existing_profile_consumer import validate_inner, validate_owner
from pilot_inputs import compact, require, resolve, sha
from pilot_observation import evaluate, wire_identity_and_first_text
from profile_summary import summarize
from split_admission import validate_before

RUN = "i03-deep-c4-short-http-v1"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def load(path, cap=16*1024**2):
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(len(raw) <= cap, "private artifact exceeds reader bound")
    return raw


def one(items, predicate, *, optional=False):
    rows = [row for row in items if predicate(row)]
    require(len(rows) == 1 or optional and not rows, "ambiguous or missing original receipt")
    return rows[0] if rows else None


def consume(folder, records, terminal, host_root, host_shared, spec, *, decode_sse, grade):
    step = terminal["step"]
    require(terminal["state"].split()[0] in TERMINAL and step.startswith("1087656.")
            and terminal["label"] == RUN, "unconfirmed original terminal step")
    before = one(records, lambda row: row.get("controller") == "before")["snapshot"]
    leased = one(records, lambda row: row.get("kind") == "i03_device_snapshot")
    require(leased["step"] == step and leased["holder_scan_location"] == "slurm_host_before_container"
            and leased["before_sha256"] == sha(json.dumps(before,sort_keys=True,separators=(",",":")).encode()),
            "wrong device lease")
    validate_before(before, leased)
    post_receipt = one(records, lambda row: row.get("controller") == "post", optional=True)
    post = post_receipt["snapshot"] if post_receipt else None
    clean = bool(post and post["kind"] == "i03_device_snapshot" and post["mode"] == "cpu"
                 and post["node"] == leased["node"] and post["boot_id"] == leased["boot_id"]
                 and post["pci"] == leased["pci"] and post["kfd_holder_count"] == 0
                 and post["step"].startswith("1087656.")
                 and int(post["step"].split(".")[1]) > int(step.split(".")[1])
                 and post["monotonic_ns"] > leased["monotonic_ns"]
                 and post["accounting_state"] == "COMPLETED" and post["exit_code"] == "0:0")
    result = dict(step=step, terminal=terminal, before_step=before["step"],
                  post_step=post["step"] if post else None, independent_post_clean=clean,
                  planned=3, normal_service_completion=False)
    complete = one(records, lambda row: row.get("kind") == "i03_r454_http"
                   and row.get("stage") == "complete", optional=True)
    if complete is None:
        progress = folder / "progress.json"
        if progress.is_file():
            rows = json.loads(load(progress))
            require(len(rows) == 3 and all(row["case_id"] == "gsm8k-r444-739" for row in rows),
                    "foreign progress slots")
            result["slots"] = [row["state"] for row in rows]
        return result
    require(terminal["state"] == "COMPLETED" and terminal["exit_code"] == "0:0"
            and complete["step"] == step and complete["planned"] == complete["recorded"] == 3
            and complete["normal_destroy"] is True, "unconfirmed completed service")
    pins = complete["artifacts_SHA256"]
    fixed = {"progress.json","service-config.json","service.log","profile-summary.json",
             "sparse_profile_hook.py","existing_profile_consumer.py","diagnostic_binding.py","consume.py","split_capture.py"}
    fixed |= {RUN + "/" + filename for filename in
              ("owner.json","MODEL_READY.json","ready-engine.json","closed-engine.json")}
    fixed |= {f"request-{i}{ext}" for i in range(3)
              for ext in (".wire",".clock.jsonl",".summary.json")}
    dynamic = {f"request-1-decode-{i}-{suffix}.json" for i in (10,11)
               for suffix in ("owner","inner","metadata","c4-host")}
    require(fixed <= set(pins) <= fixed | dynamic, "unexpected artifact inventory")
    for path, expected in pins.items():
        require(sha(load(folder/path)) == expected, "wrong artifact digest")
    def read(name):
        return json.loads(load(folder/RUN/name))
    owner, ready, engine, closed = map(read,("owner.json","MODEL_READY.json","ready-engine.json","closed-engine.json"))
    identity = owner["identity"]
    require(ready["identity"] == engine["identity"] == closed["identity"] == identity
            and identity["actual_step"] == step and identity["run"] == RUN
            and identity["provider_config_SHA256"] == pins["service-config.json"]
            and engine["loaded_key_count"] == 67569 and engine["capacity"] == 32768 and engine["rows"] == 1
            and ready["Model_READY"] is True and ready["Native_internal_owner_count"] == 8
            and ready["Native_SHA256"] == "95625a4f8a87ab588c06611597bd320e2b96cabf5e4928e7b783b1b59aae1d8e"
            and owner["Native_SHA256"] == ready["Native_SHA256"]
            and set(owner["actual_providers_before_weights"].values()) == {
              "/refactor/candidates/Deep-C4-short-host-split-CPU-v1/artifacts/libDeep_c4_short_host_split_v1.so"
            }
            and closed["normal"] is True and closed["engine_destroy_returned_zero"] is True
            and closed["observed_all8_threads_exited"] is True and len(owner["requests"]) == 3,
            "bad engine identity or teardown")
    (entry, _, case), _ = resolve(host_root,spec,host_shared=host_shared)
    compact_results = []
    for i in range(3):
        wire = load(folder/f"request-{i}.wire")
        reads = [json.loads(line) for line in load(folder/f"request-{i}.clock.jsonl").splitlines()]
        rid, _ = wire_identity_and_first_text(wire,reads)
        raw = one(owner["requests"],lambda row:row["request_id"] == rid)
        require(raw["r454_index"] == i,"wrong owner occurrence")
        observed = evaluate(entry,case,wire,reads,raw,decode_sse=decode_sse,grade=grade)
        require(compact(observed) == compact(json.loads(load(folder/f"request-{i}.summary.json"))),
                "request metric recomputation differs")
        observed.pop("request_id")
        observed.pop("backend_ITL_s")
        compact_results.append(observed)
    phases, files = summarize(owner["requests"],folder)
    require(set(pins) == fixed | set(files) and phases["qualified_samples"] == complete["qualified_profile_samples"]
            and compact(phases) == compact(json.loads(load(folder/"profile-summary.json"))),
            "phase metric recomputation differs")
    config_raw = load(host_shared/"hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json")
    require(sha(config_raw) == spec["processor_files_SHA256"]["config.json"], "wrong model configuration")
    config = json.loads(config_raw)
    ratios = [int(ratio) for ratio in config["compress_ratios"][:config["num_hidden_layers"]]]
    require(len(ratios) == 43 and [ratios.count(i) for i in (0,128,4)] == [2,20,21],
            "wrong exact cache layout")
    kinds = [{0:0,128:1,4:2}[ratio] for ratio in ratios]
    for index in (10,11):
        stem = f"request-1-decode-{index}"
        if stem + "-inner.json" not in pins:
            continue
        prior = validate_owner(load(folder/(stem+"-owner.json")),1)
        inner = validate_inner(load(folder/(stem+"-inner.json")),kinds,1,1)
        bindings = {(x["device"],x["layer"]):(x["owner"],x["stream"])
                    for x in prior["raw_owner_events"]["rows"]}
        require(all(bindings[(x["device"],x["layer"])] == (x["owner"],x["stream"])
                    for x in inner["raw_inner_events"]["rows"]), "observer packet owner identity differs")
    result.update(normal_service_completion=True, requests=compact_results, profiles=phases,
                  artifacts_SHA256=pins, exclusive_gpu_busy_measured=False,
                  user_decode_measured=False, broad_quality_measured=False)
    return result
