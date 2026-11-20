"""Independently verify the Qwen LoRA one versus two layers per checkpoint."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

REQUEST = "6231603e7ad54f979274757fdf827a95"
PACKAGE = "fleet-qwen-lora-paired-layer-checkpoint-v1"
MANIFEST_SHA256 = "4f6f724333706339c77721c6abb78bfff9025b341e6b8d88835e786c5489d5fc"
PARENTS = {"1087656": "1087656.494", "1103606": "1103606.398"}
PILOT = "1103606.397"
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results") / f"{PACKAGE}-{REQUEST}"
ORDER = (1, 2, 2, 1)
SAMPLE_OFFSETS = (45, 55, 65, 80, 100, 125, 155, 200, 260, 340, 390)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def summarize(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values))


def read(path, pins):
    with path.open("rb") as stream:
        raw = stream.read(65537)
    need(0 < len(raw) <= 65536, "ORIGINAL_RECEIPT_SIZE")
    pins[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def account():
    expected = {step: (f"qwen-lora-paired-layer-{REQUEST}-{parent}", 296)
                for parent, step in PARENTS.items()}
    expected[PILOT] = (f"qwen-lora-paired-layer-pilot-{REQUEST}", 8)
    groups = {}
    for parent in PARENTS:
        command = ["sacct", "-nP", "-j", parent,
                   "--format=JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%160"]
        raw = subprocess.check_output(command, text=True, timeout=12)
        for line in raw.splitlines():
            row = line.split("|")
            if row[0] not in expected:
                continue
            label, count = expected[row[0]]
            need(len(row) == 6 and row[0] not in groups and row[1] == label
                 and f"gres/dcu={count}" in row[5].split(","), "ORIGINAL_ACCOUNT_IDENTITY")
            groups[row[0]] = row
    return groups


def review_card(card, epoch, node, step, rank, schedule):
    need(card["node"] == node and card["step"] == step and card["card"] == rank
         and card["launch_epoch"] == epoch and card["error"] is None and card["pid"] > 0,
         "CARD_OWNER_OR_ERROR")
    result, = card["results"]
    need(result["rows_by_update"] == schedule and result["parameter_tensors"] == 504
         and result["parameter_elements"] == 43646976 and result["microbatches"] == 4
         and result["timed_steps_per_arm"] == 20
         and result["synthetic_operator_training_only"] is True,
         "WORKLOAD_BINDING")
    arms = result["caps"]
    need(tuple(x["layers_per_checkpoint"] for x in arms) == ORDER, "ARM_ORDER")
    for arm in arms:
        expected_calls = 36 // arm["layers_per_checkpoint"] * 4 * 24
        need(arm["checkpoint_early_stop"] is True and arm["actual_forward_checkpoint_calls"] == expected_calls
             and arm["checkpoint_preserve_rng_state"] is True and arm["gradient_as_bucket_view"] is True
             and arm["bucket_cap_mib"] == 25 and 0 < arm["distinct_live_gradient_storages"] < 504,
             "CHECKPOINT_AND_DDP_BINDING")
        need(len(arm["numeric"]) == 2, "NUMERIC_CARDINALITY")
        for row in arm["numeric"]:
            need(len(row) == 9 and all(math.isfinite(v) and v >= 0 for v in row)
                 and row[-1] > 0 and max(row[:2]) <= 5e-5 and max(row[2:4]) <= 5e-6 and max(row[4:6]) <= 5e-5,
                 "NUMERIC_DELTA")
        for key in ("timed_steps", "timed_optimizer_seconds", "steady_peak_allocated_bytes"):
            vals = arm[key]
            need(len(vals) == 20 and all(math.isfinite(v) and v > 0 for v in vals),
                 "INVALID_PAIRED_MEASUREMENT")
    return arms


def audit():
    rows = account()
    terminal_path = ROOT / "dispatch-terminal.json"
    result = dict(original_request=REQUEST, observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  accounting=rows, formal_training=False, planned_cards=592)
    if not terminal_path.is_file() or any(row[2] in {"RUNNING","PENDING","COMPLETING"} for row in rows.values()):
        return dict(result, state="ACTIVE_OR_ACCOUNTING_PENDING")
    pins = {}
    dispatch = read(terminal_path, pins)
    intent = read(ROOT/"intent.json", pins)
    launcher = read(ROOT/"launcher-owner.json", pins)
    pilot = read(ROOT/"pilot-qualified.json", pins)
    epoch = intent["epoch"]
    need(all(x["request"] == REQUEST for x in (intent, dispatch, launcher))
         and launcher["source_manifest_SHA256"] == MANIFEST_SHA256
         and launcher["epoch"] == dispatch["epoch"] == epoch
         and launcher["controller_PID"] > 0, "DISPATCH_OWNER_OR_SCOPE")
    need(rows[PILOT][:4] == [PILOT, pilot["accounting"][1], "COMPLETED", "0:0"]
         and pilot["receipt"]["post_clean"] is True, "PILOT_CONTRADICTION")
    if set(rows) != {*PARENTS.values(), PILOT} or any(row[2:4] != ["COMPLETED","0:0"] for row in rows.values()):
        return dict(result, state="FAILED_OR_ACCOUNTING_INCOMPLETE", source_terminal=dispatch)
    need(not dispatch["errors"] and dispatch["returncodes"] == {parent: 0 for parent in PARENTS}, "CONTROLLER_FAILURE")
    expected_nodes = intent["parents"]
    need(set(expected_nodes) == set(PARENTS) and all(len(nodes) == 37 for nodes in expected_nodes.values())
         and len({n for group in expected_nodes.values() for n in group}) == 74, "EXACT_NODE_COUNT")

    sys.path.insert(0, str(ROOT))
    from qualified_sampler import aggregate, node_map
    ledger = node_map(read(ROOT/"ledger.json", pins))
    metadata = read(ROOT/"ROW_METADATA.json", pins)["rows_by_rank_and_update"]
    selected = {node: ledger[node] for nodes in expected_nodes.values() for node in nodes}
    device_pins = {}
    end_to_end, optimizer, peak = [], [], []
    largest = [0.0] * 6
    for parent, nodes in expected_nodes.items():
        step = PARENTS[parent]
        for node in nodes:
            need(ledger[node][0] == parent, "WRONG_PARENT_NODE")
            path = ROOT/"nodes"/node
            inv = read(path/"inventory.json", pins)
            term = read(path/"terminal.json", pins)
            need(all(x["node"] == node and x["step"] == step and x["launch_epoch"] == epoch for x in (inv,term))
                 and term["error"] is None and term["post_kfd_holders"] == 0
                 and term["exits"] == {str(i): 0 for i in range(8)}, "ORIGINAL_NODE_CLEANUP")
            pcis = set()
            baseline, candidate = [], []
            for rank in range(8):
                card = read(path/f"card-{rank}.json", pins)
                pcis.add(card["pci"])
                arms = review_card(card, epoch, node, step, rank, metadata[rank])
                for arm in arms:
                    for row in arm["numeric"]:
                        largest = [max(a,b) for a,b in zip(largest,row[:6],strict=True)]
                if not baseline:
                    for index, arm in enumerate(arms):
                        (candidate if ORDER[index] == 2 else baseline).append(arm)
                peak.append(max(max(arms[i]["steady_peak_allocated_bytes"]) for i in (1,2))
                            - max(max(arms[i]["steady_peak_allocated_bytes"]) for i in (0,3)))
            device_pins[node] = list(inv["PCI_KFD"])
            need(pcis == set(device_pins[node]) and len(pcis) == 8, "CARD_PCI_BIJECTION")
            for key, target in (("timed_steps",end_to_end),("timed_optimizer_seconds",optimizer)):
                target.append(statistics.median(candidate[0][key] + candidate[1][key])
                              / statistics.median(baseline[0][key] + baseline[1][key]))
    samples = {}
    for offset in SAMPLE_OFFSETS:
        data = {}
        for node in selected:
            path = ROOT/"nodes"/node/f"busy-{offset}.json"
            if path.is_file():
                row = read(path, pins)
                need(row["launch_epoch"] == epoch and row["step"] == PARENTS[selected[node][0]], "BUSY_ORIGINAL_SCOPE")
                data[node] = row
        if data:
            metrics = aggregate(selected, device_pins, data, intent["sample_epoch"]+offset)
            samples[offset] = {key:metrics[key] for key in ("sample_count","observed_DCU","observed_devices_mean_busy_percent")}
    identity = hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()
    return dict(result, state="PAIRED_LAYER_CHECKPOINT_TERMINAL_INDEPENDENTLY_READ", validated_cards=len(peak),
                validated_model_updates=0, matched_ddp_groups=len(end_to_end), source_receipts=len(pins), actual_forward_checkpoints_per_microbatch={"baseline":36,"candidate":18},
                group_updates=len(end_to_end)*80, steady_peak_delta_bytes=summarize(peak),
                total_update_candidate_over_baseline=summarize(end_to_end),
                unchanged_optimizer_device_event_candidate_over_baseline=summarize(optimizer),
                max_absolute=dict(zip(("param","grad","exp_avg","exp_avg_sq","summed_loss","preclip_norm"),largest,strict=True)),
                worker_reported_clean_nodes=len(device_pins), post_step_node_holders_independently_checked=False,
                original_source_receipt_digest=identity, busy=samples)


if __name__ == "__main__":
    print(json.dumps(audit(), allow_nan=False))
