"""Receiver for one explicit .1318 paired comparison; reads original media in place."""

import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time


def authority_module(raw):
    from types import ModuleType
    value = ModuleType("frozen_1318_target_authority")
    exec(compile(raw, "<paired authority>", "exec"), value.__dict__)
    return value


def preflight(packet):
    auth = authority_module(packet["auxiliary"]["authority.py"])
    evidence = auth.cpu_evidence(packet["public_cpu"], packet["cpu_terminal"])
    selected = auth.selected_head(evidence, packet["training"])
    original, available, missing = auth.verified_dev24()
    historical = {number: auth.prior_pair(number, pinned, packet["candidate_execution_sha"])
                  for number, pinned in packet["prior"].items()}
    matches = [number for number, past in historical.items() if auth.compatible(selected, past, packet["candidate_execution_sha"])]
    common = {"head_file_SHA256": selected["file_SHA256"], "head_module_SHA256": selected["module_SHA256"],
              "approved_scope": packet["scope"],
              "source_actual_CPU_step": evidence["actual_step"], "source_actual_CPU_provider_SHA256": auth.PROVIDER_SHA,
              "inputs_SHA256": auth.INPUT_SHA, "base": auth.BASE, "windows": 24, "inputs": 22, "original_missing": 2,
              "past_comparisons": {number: {key: values[key] for key in ("gpu_step", "file_SHA256", "module_SHA256",
                                   "inputs_SHA256", "base", "scorer_full_SHA256", "scorer_execution_SHA256")}
                                   for number, values in historical.items()}}
    if matches:
        return auth, dict(common, state="REUSE_ORIGINAL_EQUIVALENT_WINDOW_RESULTS",
                          reuse=auth.reuse_rows(matches[0], available), new_GPU=0), None
    return auth, dict(common, state="EXACT_FROZEN_PAIR_IDENTITY_READBACK", new_GPU=0), (original, selected, missing)


def save(path, raw):
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())


def prepare_actions(packet, auth, prepared):
    scope = packet["scope"]
    request, parent, node = [scope[k] for k in ("request", "parent", "node")]
    label = "vg-1318-dev24-" + request
    epoch = packet["epoch"]
    auth.need(packet["label"] == label and re.fullmatch(r"[a-f0-9]{32}", request)
              and scope["approved_operational_scope"] is scope["only_exclusive_node"] is True
              and packet["deadline"] == epoch + 1800 and epoch <= time.time() < epoch + 100
              and re.fullmatch(r"[a-f0-9]{40}", packet["publication"]), "EXACT_ACTIVE_APPROVED_SINGLE_EPOCH")
    stage = auth.RESULTS / label
    installed = auth.PROJECT / "incoming" / label
    auth.need(not stage.exists() and not stage.is_symlink() and not installed.exists() and not installed.is_symlink(),
              "ONE_ORIGINAL_REQUEST_WITHOUT_REPLAY")
    sources = packet["sources"]
    raw_index = sources[auth.LEGACY_INDEX]
    auth.need(auth.sha(raw_index) == packet["source_index_SHA256"], "PINNED_COMPUTE_SOURCES")
    index = json.loads(raw_index)
    auth.need(len(index) == 16 and set(sources) == {*index, auth.LEGACY_INDEX}, "SOURCE_ONLY_CLOSURE")
    for relative, raw in sources.items():
        path = PurePosixPath(relative)
        auth.need(not path.is_absolute() and ".." not in path.parts and path.suffix in (".py", ".json")
                  and 0 < len(raw) <= 2**21 and (relative == auth.LEGACY_INDEX or auth.sha(raw) == index[relative]),
                  "PINNED_SOURCE_ONLY_CLOSURE")
    original, head, missing = prepared
    installed.mkdir(mode=0o700)
    for relative, raw in sources.items():
        file = installed / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        save(file, raw)
    stage.mkdir(mode=0o700)
    heads = auth.encode({auth.ARM: head})
    save(stage / "head-authority.json", heads)
    record = dict(original, original_stage_SHA256=auth.DEV_STAGE_SHA,
                  head_authority_SHA256=auth.sha(heads), pair_reader_SHA256=index[auth.READER],
                  source_3090_result_SHA256=auth.CPU_RESULT_SHA, single_epoch=epoch,
                  original_failed_windows=missing, whole_deadline=packet["deadline"], approved_placement=scope)
    record.pop("scorer_arguments", None)
    raw_stage = auth.encode(record)
    save(stage / "stage-complete.json", raw_stage)
    argv = [sys.executable, "-B", str(installed / "tools/control_vggt_tum_cross_camera_dev.py"),
            str(stage), "--stage-sha", auth.sha(raw_stage), "--request", request,
            "--publication", packet["publication"], "--manifest-sha", packet["source_index_SHA256"]]
    return argv


def run(packet):
    os.umask(0o077)
    auth, readback, prepared = preflight(packet)
    print(json.dumps(readback, sort_keys=True), flush=True)
    if packet.get("independent_preflight_only") or prepared is None:
        return 0
    argv = prepare_actions(packet, auth, prepared)
    timeout = min(1500, packet["epoch"] + 1500 - time.time(), packet["deadline"] - time.time())
    auth.need(timeout > 1320, "FRESH_ENOUGH_FOR_ORIGINAL_PRE_GPU_POST")
    return subprocess.run(argv, timeout=timeout).returncode


if __name__ == "__main__":
    raise SystemExit(run(INPUT))
