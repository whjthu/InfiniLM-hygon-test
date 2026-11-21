"""Close the actual 73-node Graph32/64 v5 wave from independent original receipts."""

import builtins
from hashlib import sha256
import json
import math
import statistics
from pathlib import Path
from types import ModuleType

from prepare_training80_vggt_direct_finish import bounded
from report_training80_backlog import REPO, TRAINING80_LEDGER, hex_value, need


REQUEST = "ec0e9bff03066610f973fe9180a40d56"
PACKAGE = Path("code/bw1000/fleet-bf16-projection-graph-depth-staggered-v5")
DIRECTORY = Path("local-only") / ("O01-resident-BF16-graph-depth-staggered-v5-" + REQUEST + "-original")
EPOCH = 1795302141.230279
PLAN_SHA = "b338440f15a52f5f5c4e21ec7935dd8d3d3802e326ac3310db39540877e1c9cd"
CLAIM_SHA = "d5815497748036023c0c522408783e427ba46dbbe7b4211cea8bfb00ad6e433d"
SCOPE_SHA = "444ced1ffc4bf5231b7fcf7a2410e95a3cfd99370b0e015614bf4c2aa301ea07"
CONTROLLER_SHA = "e914068b5a1f1574ef180d4e6f49b9e6a44025ea4487d935713f163491ed6432"
MANIFEST_SHA = "7473271a1f2fb0ef7a7f82425dd023452f218122eddd684332785c0e8af2afc7"
ENVELOPE_SHA = "3d288fcafca92495993eab494f6534dee5237e1a6a03d526c1725a7e0604b562"
PUBLICATION = "0a7354b95bcc3a272b7e63666cab695a5340f51d"
CASE = "O01.resident-bf16-graph-depth-staggered"
DENOMINATOR = dict(planned_nodes=73, planned_cards=584, planned_arms=2920, planned_busy_card_samples=35040)
NANO = 10**9
VERIFIER_SHA = "d72afea2a45196f186adc00ff8aaa47895c983b1bcb0d18fc0c7fae56cb2cc88"
ORDER = ("graph64", "graph32", "graph32", "graph64", "common64")
WORK_FIELDS = ("complete_cards", "measured_partial_cards", "measured_card_samples", "groups", "gemm_calls",
               "useful_rows")


def sha(raw):
    return sha256(raw).hexdigest()


def frozen_source(repo):
    claim = bounded(repo / PACKAGE / "ROOT_CLAIM.json", 8192)
    manifest = bounded(repo / PACKAGE / "SOURCE_FILES.json", 16384)
    need(sha(claim) == CLAIM_SHA and json.loads(claim)["request"] == REQUEST and sha(manifest) == MANIFEST_SHA,
         "original Graph64 v5 claim or manifest differs")
    files = {"SOURCE_FILES.json": manifest}
    for name, spec in json.loads(manifest)["files"].items():
        relative = Path(spec["path"])
        need(not relative.is_absolute() and ".." not in relative.parts, "invalid v5 source path")
        path = repo / relative
        need(path.resolve(strict=True) == path, "noncanonical v5 source member")
        raw = bounded(path, 32768)
        need((len(raw), sha(raw)) == (spec["bytes"], spec["SHA256"]), "original v5 source member differs: " + name)
        files[name] = raw
    identity = sha(json.dumps({name: sha(data) for name, data in sorted(files.items())},
                              separators=(",", ":")).encode())
    need(identity == ENVELOPE_SHA, "v5 source differs from the actual original envelope")
    return files


def scope_loader(raw, filename, scope):
    # Resolve the sole staged local import without changing the caller's modules or import paths.
    contract_path = Path(filename).with_name("contract.py")
    contract = ModuleType("_graph64_v5_exact_contract")
    contract_raw = bounded(contract_path, 32768)
    need(sha(contract_raw) == "9c66ea595290fdcaa449c311b478352db72bdf9529b125bce0156ba03b190e22",
         "frozen v5 measurement contract differs")
    exec(compile(contract_raw, str(contract_path), "exec"), vars(contract))

    def import_staged(name, *args, **kwargs):
        return contract if name == "contract" else builtins.__import__(name, *args, **kwargs)

    scope["__builtins__"] = dict(vars(builtins), __import__=import_staged)
    exec(compile(raw, filename, "exec"), scope)


def empty_arm(mode):
    return dict(mode=mode, **dict.fromkeys(WORK_FIELDS, 0), complete_groups_by_quarter=[0]*4)


def add_arm(out, source):
    for key in WORK_FIELDS:
        out[key] += source[key]
    out["complete_groups_by_quarter"] = [a+b for a, b in zip(
        out["complete_groups_by_quarter"], source["complete_groups_by_quarter"], strict=True)]


def validate_node_work(node, step):
    arms = node.get("per_arm", [])
    need(type(arms) is list and len(arms) in {0, len(ORDER)}, "invalid sustained node arm set")
    if not arms:
        need(node["valid_cards"] == node["completed_arms"] == node["actual_busy_card_samples"] == 0,
             "v5 worker totals lack independent per-arm records")
        return ()
    need(node["original_step"] == step and type(node["cooperative_stopped"]) is bool
         and all(hex_value(node["raw_SHA256"][name], 64) for name in ("intent.json", "terminal.json")),
         "v5 partial worker does not bind the original numbered step")
    for mode, arm in zip(ORDER, arms, strict=True):
        need(arm["mode"] == mode and arm["planned_cards"] == 8
             and all(type(arm[key]) is int and arm[key] >= 0 for key in WORK_FIELDS),
             "v5 arm scope or measured work counter differs")
        complete, partial, groups, gemms = (arm[key] for key in ("complete_cards", "measured_partial_cards",
                                                               "groups", "gemm_calls"))
        quarters = arm["complete_groups_by_quarter"]
        depth = 32 if mode == "graph32" else 64
        need(complete+partial <= 8 and 0 <= arm["measured_card_samples"] <= 12*(complete+partial)
             and (partial == 0 or node["cooperative_stopped"] is True)
             and complete+partial <= groups and (groups == 0 or complete+partial > 0)
             and gemms == depth*groups and arm["useful_rows"] == 186*gemms
             and len(quarters) == 4 and all(type(n) is int and n >= 0 for n in quarters) and sum(quarters) == groups,
             "v5 graph depth/useful rows, partial cards, samples or quarter denominator differs")
    need(sum(arm["complete_cards"] for arm in arms) == node["completed_arms"]
         and sum(arm["measured_card_samples"] for arm in arms) == node["actual_busy_card_samples"]
         and min(arm["complete_cards"] for arm in arms) >= node["valid_cards"],
         "v5 independent node and arm totals disagree")
    return arms


def node_pairs(node):
    pairs = node["per_card_graph32_over_graph64_pairs"]
    need(type(pairs) is dict and len(pairs) == 8, "v5 qualified node lacks eight paired cards")
    values = []
    for pair in pairs.values():
        need(type(pair) is list and len(pair) == 2
             and all(type(value) in {int, float} and 0 < value and math.isfinite(value) for value in pair),
             "v5 Graph32 versus Graph64 pair differs")
        values.extend(pair)
    return values


def common_interval(node):
    limits = node["common_window_epoch_ns"]
    opening = round((EPOCH+375)*NANO)
    ending = round((EPOCH+435)*NANO)
    need(type(limits) is list and len(limits) == 2 and all(type(ns) is int for ns in limits)
         and opening <= limits[0] <= opening+500_000_000
         and ending-150_000_000 <= limits[1] <= ending+500_000_000,
         "v5 common64 measured interval is outside the original local clock window")
    return limits


def bound_verifier():
    path = Path(__file__).resolve().with_name("close_graph64_original_backfill.py")
    source = bounded(path, 65536)
    need(sha(source) == VERIFIER_SHA, "frozen Graph64 per-shard verifier bytes differ")
    for before, after in (
        (b'len(plan["selected"]) == 75', b'len(plan["selected"]) == DENOMINATOR["planned_nodes"]'),
        (b'31-shard or 75-node scope', b'31-shard or 73-node v5 scope'),
        (b'exec(compile(scope_raw, str(scope_file), "exec"), scope)',
         b'scope_loader(scope_raw, str(scope_file), scope)'),
        (b'"completed_arms": 32, "actual_busy_card_samples": 128',
         b'"completed_arms": 40, "actual_busy_card_samples": 480'),
        (b'<= 140 and all(', b'<= 470 and all('),
    ):
        need(source.count(before) == 1, "frozen Graph64 proof binding differs")
        source = source.replace(before, after)
    checker = {"__name__": "_graph64_v5_original_verifier", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), checker)
    checker.update(REQUEST=REQUEST, PACKAGE=PACKAGE, DIRECTORY=DIRECTORY, PLAN_SHA=PLAN_SHA,
                   CLAIM_SHA=CLAIM_SHA, SCOPE_SHA=SCOPE_SHA, CONTROLLER_SHA=CONTROLLER_SHA,
                   PUBLICATION=PUBLICATION, CASE=CASE, DENOMINATOR=DENOMINATOR, scope_loader=scope_loader)
    input_reader, check_counts, close_shards = (checker[name] for name in ("original_inputs", "counters", "close"))

    def original_inputs(repo):
        originals = input_reader(repo)
        frozen_source(repo)
        need(originals[0]["source_SHA256"] == ENVELOPE_SHA and originals[0]["campaign_epoch"] == EPOCH,
             "v5 original plan does not bind the deployed envelope and exact launch epoch")
        return originals

    def counters(verdict, unit, item, row):
        passed, counts = check_counts(verdict, unit, item, row)
        totals = [empty_arm(mode) for mode in ORDER]
        cooperative, incomplete, comparable, intervals = [], [], {}, {}
        for name in unit["nodes"]:
            node = verdict["nodes"].get(name)
            if node is None or node.get("passed") is not True:
                incomplete.append(name)
            if node is None:
                continue
            arms = validate_node_work(node, row[0])
            if not arms:
                continue
            if node["cooperative_stopped"]:
                cooperative.append(name)
            for total, arm in zip(totals, arms, strict=True):
                add_arm(total, arm)
            if node.get("passed") is True:
                need(not node["cooperative_stopped"] and node["valid_cards"] == 8
                     and node["completed_arms"] == 40 and node["actual_busy_card_samples"] == 480,
                     "node pass contradicts its complete Graph32/64 arms")
                intervals[name] = common_interval(node)
                if passed:
                    comparable[name] = node_pairs(node)
        need(not passed or not incomplete, "a cooperative or incomplete node prevents full shard qualification")
        return passed, dict(counts, per_arm_work=totals, cooperative_stop_nodes=cooperative,
                            incomplete_worker_nodes=incomplete, local_common_windows_epoch_ns=intervals,
                            qualified_pair_ratios=comparable)

    def close(connection, originals, reconcile_raw, capture_raw, snapshot_raw, post_raw, **kwargs):
        verdict = json.loads(reconcile_raw)
        need(verdict["campaign_epoch"] == EPOCH == originals[0]["campaign_epoch"]
             and verdict["clock_inter_node_bounded"] is False
             and verdict["common_window_inter_node_overlap_ns"] is None,
             "original v5 epoch or independently unbounded clock contract differs")
        result = close_shards(connection, originals, reconcile_raw, capture_raw, snapshot_raw, post_raw, **kwargs)
        totals = [empty_arm(mode) for mode in ORDER]
        cooperative, incomplete, missing = set(), set(), set()
        local_intervals, pairs = {}, {}
        for shard in result["shards"].values():
            payload = shard.get("result", shard)
            cooperative.update(payload.get("cooperative_stop_nodes", []))
            incomplete.update(payload.get("incomplete_worker_nodes", []))
            if "incomplete_worker_nodes" not in payload:
                missing.update(shard["nodes"])
            local_intervals.update(payload.get("local_common_windows_epoch_ns", {}))
            pairs.update(payload.get("qualified_pair_ratios", {}))
            for total, arm in zip(totals, payload.get("per_arm_work", ()), strict=False):
                add_arm(total, arm)
        ratios = [ratio for node_pairs in pairs.values() for ratio in node_pairs]
        result.update(verified_per_arm_work=totals, cooperative_stop_nodes=sorted(cooperative),
                      incomplete_worker_nodes=sorted(incomplete), worker_evidence_unverified_nodes=sorted(missing),
                      local_common_intervals=len(local_intervals), planned_common_nodes=DENOMINATOR["planned_nodes"],
                      common_clock_inter_node_bounded=False, common_window_inter_node_overlap_ns=None,
                      paired_comparison=dict(qualified_nodes=len(pairs), measured_pairs=len(ratios),
                                             planned_pairs=DENOMINATOR["planned_cards"]*2,
                                             median_graph32_over_graph64=statistics.median(ratios) if ratios else None))
        return result

    checker.update(original_inputs=original_inputs, counters=counters, close=close)
    return checker


_checker = bound_verifier()
terminal_receipt = _checker["terminal_receipt"]
original_inputs = _checker["original_inputs"]
close = _checker["close"]


if __name__ == "__main__":
    _checker["main"]()
