"""Close the actual 74-node Graph64 v4 wave, retaining verified partial work."""

import builtins
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType

from prepare_training80_vggt_direct_finish import bounded
from report_training80_backlog import REPO, TRAINING80_LEDGER, hex_value, need


REQUEST = "2da5367b3f4cd82011adb7c56b7c7cd9"
PACKAGE = Path("code/bw1000/fleet-bf16-projection-graph64-sustained-v4")
DIRECTORY = Path("local-only") / ("O01-resident-BF16-graph64-sustained-v4-" + REQUEST + "-original")
PLAN_SHA = "ddad9bdca507e6099617f6cbe3da949de07b9687f9a87c635af76a425566a4f1"
CLAIM_SHA = "310face134b1617e9c077f9733fa2be22e68e67e237f9b7280d456e79f03a61a"
SCOPE_SHA = "0ecb4c3d77df59e5d8fb4d59bf879e6cb55032b2c1fa7b91d66fd63f1affa445"
CONTROLLER_SHA = "3ffeec0233b77a0f5a3537ad2befbf495323dac823772581a2e2d87c3663e1c8"
MANIFEST_SHA = "0e232509f468d85caefc9276f236535646a6b20f69813a10f930a02ffaded35d"
ENVELOPE_SHA = "bf477cd55e9cddcabadb87a8dcc7b072cb5bf70d0405d72526d8850e973f0c6e"
PUBLICATION = "776f4e2495272f0eb0cfc26c969896a52c84c8d9"
CASE = "O01.resident-bf16-graph64-sustained-t01-geometry"
DENOMINATOR = dict(planned_nodes=74, planned_cards=592, planned_arms=2368, planned_busy_card_samples=28416)
VERIFIER_SHA = "d72afea2a45196f186adc00ff8aaa47895c983b1bcb0d18fc0c7fae56cb2cc88"
ORDER = ("native186", "padded192", "padded192", "native186")
WORK_FIELDS = ("complete_cards", "measured_partial_cards", "measured_card_samples", "groups", "gemm_calls",
               "useful_rows", "physical_rows", "padded_rows")


def sha(raw):
    return sha256(raw).hexdigest()


def frozen_source(repo):
    claim = bounded(repo / PACKAGE / "ROOT_CLAIM.json", 8192)
    manifest = bounded(repo / PACKAGE / "SOURCE_FILES.json", 16384)
    need(sha(claim) == CLAIM_SHA and json.loads(claim)["request"] == REQUEST and sha(manifest) == MANIFEST_SHA,
         "original Graph64 v4 claim or manifest differs")
    files = {"SOURCE_FILES.json": manifest}
    for name, spec in json.loads(manifest)["files"].items():
        relative = Path(spec["path"])
        need(not relative.is_absolute() and ".." not in relative.parts, "invalid v4 source path")
        path = repo / relative
        need(path.resolve(strict=True) == path, "noncanonical v4 source member")
        raw = bounded(path, 32768)
        need((len(raw), sha(raw)) == (spec["bytes"], spec["SHA256"]), "original v4 source member differs: " + name)
        files[name] = raw
    identity = sha(json.dumps({name: sha(data) for name, data in sorted(files.items())},
                              separators=(",", ":")).encode())
    need(identity == ENVELOPE_SHA, "v4 source differs from the actual original envelope")
    return files


def scope_loader(raw, filename, scope):
    # Resolve the sole staged local import without changing the caller's modules or import paths.
    contract_path = Path(filename).with_name("contract.py")
    contract = ModuleType("_graph64_v4_exact_contract")
    contract_raw = bounded(contract_path, 32768)
    need(sha(contract_raw) == "bcb3b29038d6b7d7cfa60b5ed4393cdd320b2ee92d82c58aeafb971826044030",
         "frozen v4 measurement contract differs")
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
    need(type(arms) is list and len(arms) in {0, 4}, "invalid sustained node arm set")
    if not arms:
        need(node["valid_cards"] == node["completed_arms"] == node["actual_busy_card_samples"] == 0,
             "v4 worker totals lack independent per-arm records")
        return ()
    need(node["original_step"] == step and type(node["cooperative_stopped"]) is bool
         and all(hex_value(node["raw_SHA256"][name], 64) for name in ("intent.json", "terminal.json")),
         "v4 partial worker does not bind the original numbered step")
    for mode, arm in zip(ORDER, arms, strict=True):
        need(arm["mode"] == mode and arm["planned_cards"] == 8
             and all(type(arm[key]) is int and arm[key] >= 0 for key in WORK_FIELDS),
             "v4 arm scope or measured work counter differs")
        complete, partial, groups, gemms = (arm[key] for key in ("complete_cards", "measured_partial_cards",
                                                               "groups", "gemm_calls"))
        quarters = arm["complete_groups_by_quarter"]
        physical_m = 186 if mode == "native186" else 192
        need(complete+partial <= 8 and 0 <= arm["measured_card_samples"] <= 12*(complete+partial)
             and (partial == 0 or node["cooperative_stopped"] is True)
             and complete+partial <= groups and (groups == 0 or complete+partial > 0)
             and gemms == 64*groups and arm["useful_rows"] == 186*gemms
             and arm["physical_rows"] == physical_m*gemms and arm["padded_rows"] == (physical_m-186)*gemms
             and len(quarters) == 4 and all(type(n) is int and n >= 0 for n in quarters) and sum(quarters) == groups,
             "v4 physical/useful rows, partial cards, samples or quarter denominator differs")
    need(sum(arm["complete_cards"] for arm in arms) == node["completed_arms"]
         and sum(arm["measured_card_samples"] for arm in arms) == node["actual_busy_card_samples"]
         and min(arm["complete_cards"] for arm in arms) >= node["valid_cards"],
         "v4 independent node and arm totals disagree")
    return arms


def bound_verifier():
    path = Path(__file__).resolve().with_name("close_graph64_original_backfill.py")
    source = bounded(path, 65536)
    need(sha(source) == VERIFIER_SHA, "frozen Graph64 per-shard verifier bytes differ")
    for before, after in (
        (b'len(plan["selected"]) == 75', b'len(plan["selected"]) == DENOMINATOR["planned_nodes"]'),
        (b'31-shard or 75-node scope', b'31-shard or 74-node v4 scope'),
        (b'exec(compile(scope_raw, str(scope_file), "exec"), scope)',
         b'scope_loader(scope_raw, str(scope_file), scope)'),
        (b'"actual_busy_card_samples": 128', b'"actual_busy_card_samples": 384'),
        (b'<= 140 and all(', b'<= 345 and all('),
    ):
        need(source.count(before) == 1, "frozen Graph64 proof binding differs")
        source = source.replace(before, after)
    checker = {"__name__": "_graph64_v4_original_verifier", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), checker)
    checker.update(REQUEST=REQUEST, PACKAGE=PACKAGE, DIRECTORY=DIRECTORY, PLAN_SHA=PLAN_SHA,
                   CLAIM_SHA=CLAIM_SHA, SCOPE_SHA=SCOPE_SHA, CONTROLLER_SHA=CONTROLLER_SHA,
                   PUBLICATION=PUBLICATION, CASE=CASE, DENOMINATOR=DENOMINATOR, scope_loader=scope_loader)
    input_reader, check_counts, close_shards = (checker[name] for name in ("original_inputs", "counters", "close"))

    def original_inputs(repo):
        originals = input_reader(repo)
        frozen_source(repo)
        need(originals[0]["source_SHA256"] == ENVELOPE_SHA, "v4 original plan does not bind the deployed envelope")
        return originals

    def counters(verdict, unit, item, row):
        passed, counts = check_counts(verdict, unit, item, row)
        totals = [empty_arm(mode) for mode in ORDER]
        cooperative = []
        for name in unit["nodes"]:
            node = verdict["nodes"].get(name)
            if node is None:
                continue
            arms = validate_node_work(node, row[0])
            if arms:
                if node["cooperative_stopped"]:
                    cooperative.append(name)
                for total, arm in zip(totals, arms, strict=True):
                    add_arm(total, arm)
        need(not passed or not cooperative, "a cooperative stop is not a full fixed-window qualification")
        return passed, dict(counts, per_arm_work=totals, cooperative_stop_nodes=cooperative)

    def close(*args, **kwargs):
        result = close_shards(*args, **kwargs)
        totals = [empty_arm(mode) for mode in ORDER]
        cooperative = set()
        for shard in result["shards"].values():
            payload = shard.get("result", shard)
            cooperative.update(payload.get("cooperative_stop_nodes", []))
            for total, arm in zip(totals, payload.get("per_arm_work", ()), strict=False):
                add_arm(total, arm)
        result.update(verified_per_arm_work=totals, cooperative_stop_nodes=sorted(cooperative))
        return result

    checker.update(original_inputs=original_inputs, counters=counters, close=close)
    return checker


_checker = bound_verifier()
terminal_receipt = _checker["terminal_receipt"]
original_inputs = _checker["original_inputs"]
close = _checker["close"]


if __name__ == "__main__":
    _checker["main"]()
